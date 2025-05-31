import os
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import sys
import time
import glob
import matplotlib.pyplot as plt
from IPython.display import FileLink
from scipy.special import expit

# --------- CONFIGURATIONS ---------
# Paths
CHECKPOINT_PATH = '/kaggle/input/models/epoch29_cosine_eer10.18.ckpt' 
TEST_FILE = '/kaggle/input/privatesvtest/prompts_sv.csv' 
AUDIO_DIR = '/kaggle/input/privatesvtest/audio' 
OUTPUT_FILE = '/kaggle/working/predictions.txt' 

# Model configurations - must match the trained model
EMBEDDING_DIM = 256
ENCODER_NAME = 'resnet34'
NUM_BLOCKS = 6
INPUT_LAYER = 'conv2d'
POS_ENC_LAYER_TYPE = 'abs_pos'
LOSS_NAME = 'amsoftmax'  
NUM_CLASSES = 880  

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------

print(f"Starting inference at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Using device: {DEVICE}")

# Add MFA Conformer path
sys.path.append('/kaggle/working/mfa_conformer')

# Import modules from MFA Conformer
try:
    from main import Task
    from module.dataset import load_audio
    print("Successfully imported modules from mfa_conformer")
except ImportError as e:
    print(f"Error importing modules from mfa_conformer: {e}")
    print("Please check the mfa_conformer directory")
    raise


# Create a dummy trial file
dummy_trial_path = '/kaggle/working/dummy_trial.txt'
with open(dummy_trial_path, 'w') as f:
    f.write("1 /path/to/dummy1 /path/to/dummy2\n")
# ---------------------------------------------

def extract_embeddings(model, audio_paths, device='cuda'):
    """Extract embeddings from audio files"""
    embeddings = {}
    model.to(device)
    model.eval()
    
    # Count errors for reporting
    error_count = 0
    
    for path in tqdm(audio_paths, desc="Extracting embeddings"):
        try:
            # Load audio with proper error checking
            waveform = load_audio(path, second=-1)  # Use full audio (-1)
            
            # Basic checks on the waveform
            if waveform is None or len(waveform) == 0:
                print(f"Warning: Empty audio for {path}")
                embeddings[path] = np.zeros(model.hparams.embedding_dim)
                error_count += 1
                continue
                
            # Convert to tensor and move to device
            waveform = torch.FloatTensor(waveform).unsqueeze(0).to(device)
            
            # Extract embedding
            with torch.no_grad():
                # The model might return a tuple or have different output formats
                # Make sure to get the actual embedding
                outputs = model(waveform)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    embedding = outputs[0]  # Assume first element is embedding
                else:
                    embedding = outputs
                    
                # Convert to numpy and flatten if needed
                embedding = embedding.cpu().numpy()
                if embedding.ndim > 1:
                    embedding = embedding[0]  # Get the first embedding if batch
            
            # Store embedding with audio path as key
            embeddings[path] = embedding
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            embeddings[path] = np.zeros(model.hparams.embedding_dim)
            error_count += 1
    
    # Report error rate
    if error_count > 0:
        print(f"Warning: {error_count}/{len(audio_paths)} files ({error_count/len(audio_paths)*100:.2f}%) failed to process properly")
    
    return embeddings

def compute_score(emb1, emb2):
    """Compute cosine similarity between two embeddings and normalize to [0, 1]"""
    # Normalize embeddings
    emb1 = emb1 - np.mean(emb1)
    emb2 = emb2 - np.mean(emb2)
    
    # Compute cosine similarity
    score = np.dot(emb1, emb2)
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if denom > 0:
        score = score / denom
    else:
        score = 0.0
    
    # Normalize to [0, 1]
    score = expit(score)  # Use sigmoid to normalize

    return score

def find_best_checkpoint(models_dir='/kaggle/working/models'):
    """Find the best checkpoint in the models directory based on validation metrics"""
    # Look for checkpoints with "val_eer" in filename
    best_ckpt = None
    best_eer = float('inf')
    
    # Check if there are any checkpoints with val_eer in the filename
    val_eer_ckpts = glob.glob(os.path.join(models_dir, '*val_eer*.ckpt'))
    
    if val_eer_ckpts:
        for ckpt in val_eer_ckpts:
            try:
                # Extract EER value from filename
                eer_str = ckpt.split('val_eer=')[1].split('-')[0]
                eer = float(eer_str)
                if eer < best_eer:
                    best_eer = eer
                    best_ckpt = ckpt
            except:
                continue
    
    # If no val_eer checkpoints found, try to find the last.ckpt file
    if best_ckpt is None:
        last_ckpt = os.path.join(models_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            best_ckpt = last_ckpt
    
    # If still not found, try any checkpoint file
    if best_ckpt is None:
        all_ckpts = glob.glob(os.path.join(models_dir, '*.ckpt'))
        if all_ckpts:
            # Sort by modification time (newest first)
            best_ckpt = sorted(all_ckpts, key=os.path.getmtime, reverse=True)[0]
    
    return best_ckpt

# Find best checkpoint if not specified
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Checkpoint not found at {CHECKPOINT_PATH}. Looking for best checkpoint...")
    best_checkpoint = find_best_checkpoint()
    if best_checkpoint:
        CHECKPOINT_PATH = best_checkpoint
        print(f"Using checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"No checkpoint found. Make sure you have trained a model.")
        raise FileNotFoundError("No model checkpoint found")

# Initialize model with all required parameters
print(f"Loading model from {CHECKPOINT_PATH}...")
model = Task(
    embedding_dim=EMBEDDING_DIM,
    encoder_name=ENCODER_NAME,
    num_blocks=NUM_BLOCKS,
    input_layer=INPUT_LAYER,
    pos_enc_layer_type=POS_ENC_LAYER_TYPE,
    trial_path=dummy_trial_path,
    loss_name=LOSS_NAME,  
    num_classes=NUM_CLASSES,  
    learning_rate=0.001,  
    weight_decay=0.0000001,  
    batch_size=100,  
    num_workers=2,  
    max_epochs=50,  
)

# Load trained weights
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Check if test file exists
if not os.path.exists(TEST_FILE):
    print(f"Test file not found at {TEST_FILE}")
    raise FileNotFoundError(f"Test file not found at {TEST_FILE}")

# Read test file
audio_pairs = []
with open(TEST_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"Warning: Invalid line in test file: {line.strip()}")
            continue
            
        audio1 = os.path.join(AUDIO_DIR, parts[0])
        audio2 = os.path.join(AUDIO_DIR, parts[1])
        audio_pairs.append((audio1, audio2))

print(f"Found {len(audio_pairs)} audio pairs in test file")

# Get all unique audio files
all_audio_files = []
for pair in audio_pairs:
    all_audio_files.extend(pair)
all_audio_files = list(set(all_audio_files))
print(f"Found {len(all_audio_files)} unique audio files")

# Check if audio files exist
missing_files = [path for path in all_audio_files if not os.path.exists(path)]
if missing_files:
    print(f"Warning: {len(missing_files)} audio files not found")
    if len(missing_files) < 10:
        for file in missing_files:
            print(f"  Missing: {file}")
    else:
        for file in missing_files[:5]:
            print(f"  Missing: {file}")
        print(f"  ... and {len(missing_files) - 5} more")

# Extract embeddings for all audio files
print(f"Extracting embeddings using {DEVICE}...")
embeddings = extract_embeddings(model, all_audio_files, device=DEVICE)
print(f"Extracted embeddings for {len(embeddings)} audio files")

# Compute similarity scores for each pair
print("Computing similarity scores...")
scores = []
for audio1, audio2 in tqdm(audio_pairs, desc="Computing scores"):
    emb1 = embeddings[audio1]
    emb2 = embeddings[audio2]
    score = compute_score(emb1, emb2)
    scores.append(score)

# Save scores to output file
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    for score in scores:
        f.write(f"{score:.6f}\n")

# Analyze score distribution
scores_np = np.array(scores)
print(f"Scores saved to {OUTPUT_FILE}")
print(f"Number of scores: {len(scores)}")
print(f"Score range: {np.min(scores):.6f} to {np.max(scores):.6f}")
print(f"Score mean: {np.mean(scores):.6f}")
print(f"Score standard deviation: {np.std(scores):.6f}")

# Plot score distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(scores_np, bins=50)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_plot = '/kaggle/working/score_distribution.png'
plt.savefig(output_plot)
print(f"Score distribution plot saved to {output_plot}")
plt.show()

# Display sample scores
print("\nSample of scores (first 10):")
for i, score in enumerate(scores[:10]):
    print(f"  {i+1}: {score:.6f}")

print(f"\nTesting completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Create download link for submission file
if os.path.exists(OUTPUT_FILE):
    print(f"\nPredictions file is ready for download:")
    display(FileLink(OUTPUT_FILE))
else:
    print("Predictions file not found.")