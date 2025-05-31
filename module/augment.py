import numpy as np
import torch
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import soundfile
import os
import random
from tqdm import tqdm

def compute_dB(waveform):
    """
    Tính toán chỉ số decibel cho tín hiệu âm thanh
    
    Args:
        waveform (numpy.array): Dữ liệu âm thanh đầu vào (#length).
    Returns:
        float: Giá trị decibel.
    """
    val = max(0.0, np.mean(np.power(waveform, 2)))
    dB = 10*np.log10(val+1e-4)
    return dB

class WavAugment(object):
    def __init__(self, 
                 noise_csv_path=None, 
                 gaussian_noise_prob=0.3,
                 real_noise_prob=0.3,
                 volume_prob=0.3,
                 gaussian_min_snr=10,
                 gaussian_max_snr=25,
                 real_min_snr=15,
                 real_max_snr=25,
                 volume_min=0.8,
                 volume_max=1.0005):
        """
        Khởi tạo bộ augmentation cho dữ liệu âm thanh
        
        Args:
            noise_csv_path (str, optional): Đường dẫn đến file CSV chứa đường dẫn các file noise.
            gaussian_noise_prob (float): Xác suất áp dụng nhiễu Gaussian
            real_noise_prob (float): Xác suất áp dụng nhiễu thực
            volume_prob (float): Xác suất thay đổi âm lượng
            gaussian_min_snr (float): SNR tối thiểu cho nhiễu Gaussian
            gaussian_max_snr (float): SNR tối đa cho nhiễu Gaussian
            real_min_snr (float): SNR tối thiểu cho nhiễu thực
            real_max_snr (float): SNR tối đa cho nhiễu thực
            volume_min (float): Hệ số âm lượng tối thiểu
            volume_max (float): Hệ số âm lượng tối đa
        """
        self.noise_paths = []
        self.noise_names = []
        
        # Lưu các tham số cấu hình
        self.gaussian_noise_prob = gaussian_noise_prob
        self.real_noise_prob = real_noise_prob  
        self.volume_prob = volume_prob
        self.gaussian_min_snr = gaussian_min_snr
        self.gaussian_max_snr = gaussian_max_snr
        self.real_min_snr = real_min_snr
        self.real_max_snr = real_max_snr
        self.volume_min = volume_min
        self.volume_max = volume_max
        
        if noise_csv_path is not None and os.path.exists(noise_csv_path):
            try:
                noise_df = pd.read_csv(noise_csv_path)
                self.noise_paths = noise_df["utt_paths"].values
                if "speaker_name" in noise_df.columns:
                    self.noise_names = noise_df["speaker_name"].values
                print(f"Đã nạp {len(self.noise_paths)} file nhiễu từ {noise_csv_path}")
            except Exception as e:
                print(f"Lỗi khi nạp file noise CSV: {e}")
                self.noise_paths = []
                self.noise_names = []

    def __call__(self, waveform):
        """
        Áp dụng augmentation ngẫu nhiên cho tín hiệu âm thanh
        
        Args:
            waveform (numpy.array): Tín hiệu âm thanh đầu vào.
        Returns:
            numpy.array: Tín hiệu âm thanh sau khi augment.
        """
        # Tạo một số ngẫu nhiên từ 0-9
        idx = np.random.randint(0, 10)
        
        # Xử lý theo từng trường hợp
        if idx == 0:
            waveform = self.add_gaussian_noise(waveform)
            if len(self.noise_paths) > 0:
                waveform = self.add_real_noise(waveform)

        if idx == 1 or idx == 2 or idx == 3:
            if len(self.noise_paths) > 0:
                waveform = self.add_real_noise(waveform)

        if idx == 4 or idx == 5 or idx == 6:
            waveform = self.change_volum(waveform)

        if idx == 7:
            waveform = self.change_volum(waveform)
            if len(self.noise_paths) > 0:
                waveform = self.add_real_noise(waveform)

        if idx == 8:
            waveform = self.add_gaussian_noise(waveform)

        return waveform

    def add_gaussian_noise(self, waveform):
        """
        Thêm nhiễu Gaussian vào tín hiệu âm thanh
        
        Args:
            waveform (numpy.array): Tín hiệu âm thanh đầu vào.
        Returns:
            numpy.array: Tín hiệu âm thanh với nhiễu Gaussian.
        """
        snr = np.random.uniform(low=self.gaussian_min_snr, high=self.gaussian_max_snr)
        clean_dB = compute_dB(waveform)
        noise = np.random.randn(len(waveform))
        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform

    def change_volum(self, waveform):
        """
        Thay đổi âm lượng của tín hiệu âm thanh
        
        Args:
            waveform (numpy.array): Tín hiệu âm thanh đầu vào.
        Returns:
            numpy.array: Tín hiệu âm thanh với âm lượng đã thay đổi.
        """
        volum = np.random.uniform(low=self.volume_min, high=self.volume_max)
        waveform = waveform * volum
        return waveform

    def add_real_noise(self, waveform):
        """
        Thêm nhiễu thực vào tín hiệu âm thanh từ các file noise
        
        Args:
            waveform (numpy.array): Tín hiệu âm thanh đầu vào.
        Returns:
            numpy.array: Tín hiệu âm thanh với nhiễu thực.
        """
        if len(self.noise_paths) == 0:
            return waveform
            
        clean_dB = compute_dB(waveform)

        idx = np.random.randint(0, len(self.noise_paths))
        
        try:
            # Ưu tiên sử dụng soundfile vì hỗ trợ nhiều định dạng hơn
            try:
                noise, sample_rate = soundfile.read(self.noise_paths[idx])
                noise = noise.astype(np.float64)
            except:
                sample_rate, noise = wavfile.read(self.noise_paths[idx])
                noise = noise.astype(np.float64)
                
            # Nếu nhiều kênh, chỉ lấy kênh đầu tiên
            if len(noise.shape) > 1:
                noise = noise[:, 0]
        except Exception as e:
            print(f"Lỗi khi đọc file nhiễu {self.noise_paths[idx]}: {e}")
            return waveform

        snr = np.random.uniform(self.real_min_snr, self.real_max_snr)

        noise_length = len(noise)
        audio_length = len(waveform)

        if audio_length >= noise_length:
            shortage = audio_length - noise_length
            noise = np.pad(noise, (0, shortage), 'wrap')
        else:
            start = np.random.randint(0, (noise_length-audio_length))
            noise = noise[start:start+audio_length]

        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform


def load_audio(filename, second=2):
    """
    Đọc file âm thanh và cắt/pad để có độ dài cố định
    
    Args:
        filename (str): Đường dẫn đến file âm thanh.
        second (float): Số giây cần lấy từ file âm thanh.
    Returns:
        numpy.array: Dữ liệu âm thanh đã được xử lý.
    """
    try:
        # Thử đọc với soundfile trước (hỗ trợ nhiều định dạng)
        try:
            waveform, sample_rate = soundfile.read(filename)
            if len(waveform.shape) > 1:
                waveform = waveform[:, 0]  # Lấy kênh đầu tiên nếu là stereo
        except:
            sample_rate, waveform = wavfile.read(filename)
            if len(waveform.shape) > 1:
                waveform = waveform[:, 0]  # Lấy kênh đầu tiên nếu là stereo
                
        audio_length = waveform.shape[0]
    except Exception as e:
        print(f"Error loading audio file {filename}: {e}")
        # Trả về mảng trống nếu không đọc được file
        return np.zeros(int(16000 * second)) if second > 0 else np.zeros(16000)

    if second <= 0:
        return waveform.astype(np.float64).copy()

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        start = np.int64(random.random()*(audio_length-length))
        waveform = waveform[start:start+length].astype(np.float64)
    return waveform.copy()

def augment_csv_dataset(input_csv, output_csv, augment_percentage=0.6, second=2, noise_csv_path=None):
    """
    Tạo một bản sao của dataset với 60% dữ liệu được augment
    
    Args:
        input_csv (str): Đường dẫn đến file CSV đầu vào
        output_csv (str): Đường dẫn đến file CSV đầu ra
        augment_percentage (float): Tỷ lệ dữ liệu cần augment (0.0-1.0)
        second (float): Số giây lấy từ mỗi file âm thanh
        noise_csv_path (str): Đường dẫn đến file CSV chứa noise
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Đọc dữ liệu đầu vào
        df = pd.read_csv(input_csv)
        total_samples = len(df)
        
        # Khởi tạo bộ augmentation
        augmenter = WavAugment(noise_csv_path=noise_csv_path)
        
        # Tính số lượng mẫu cần augment
        num_to_augment = int(total_samples * augment_percentage)
        
        print(f"Tiến hành augment {num_to_augment}/{total_samples} mẫu ({augment_percentage*100:.1f}%)...")
        
        # Chọn ngẫu nhiên các mẫu cần augment
        indices_to_augment = np.random.choice(total_samples, num_to_augment, replace=False)
        
        # Tạo thư mục augmented nếu chưa tồn tại
        augmented_dir = os.path.join(os.path.dirname(output_csv), "augmented_audio")
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Copy DataFrame gốc
        augmented_df = df.copy()
        
        # Augment các mẫu đã chọn
        for idx in tqdm(indices_to_augment, desc="Augmenting audio"):
            # Lấy thông tin mẫu cần augment
            path = df.iloc[idx]["utt_paths"]
            speaker = df.iloc[idx]["speaker_name"]
            label = df.iloc[idx]["utt_spk_int_labels"]
            
            # Đọc audio
            waveform = load_audio(path, second=second)
            
            # Augment audio
            augmented_waveform = augmenter(waveform)
            
            # Tạo tên file mới
            base_filename = os.path.basename(path)
            augmented_filename = f"aug_{base_filename}"
            augmented_path = os.path.join(augmented_dir, augmented_filename)
            
            # Lưu file đã augment
            try:
                soundfile.write(augmented_path, augmented_waveform, 16000)
            except:
                # Fallback to scipy if soundfile fails
                wavfile.write(augmented_path, 16000, augmented_waveform.astype(np.int16))
            
            # Cập nhật đường dẫn trong DataFrame
            augmented_df.at[idx, "utt_paths"] = augmented_path
        
        # Lưu DataFrame đã augment
        augmented_df.to_csv(output_csv, index=False)
        
        print(f"Đã tạo dataset augmented với {num_to_augment} mẫu đã được augment")
        print(f"Dataset mới đã được lưu vào {output_csv}")
        
        return True
    
    except Exception as e:
        print(f"Lỗi khi augment dataset: {e}")
        return False