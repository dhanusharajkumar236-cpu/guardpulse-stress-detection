"""Extract physiological features from raw sensor data.

Key features for stress detection:
- Heart-rate variability (HRV): temporal and frequency-domain measures
- Activity patterns: from accelerometer and motion data
- Respiratory patterns: breathing rate, amplitude
- Electrodermal activity (EDA): skin conductance levels

These features are extracted from sliding windows to create a feature matrix
for training ML models.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import pandas as pd
from typing import Tuple, List


class FeatureExtractor:
    """Extract physiological features from sensor data."""
    
    CHEST_SAMPLING_RATE = 700  # RespiBAN
    WRIST_SAMPLING_RATE = 64   # Empatica E4 (after resampling to chest rate)
    
    # Sliding window parameters
    WINDOW_SIZE_SEC = 5        # 5-second windows
    WINDOW_OVERLAP = 0.5       # 50% overlap

    def __init__(self, sampling_rate: int = 700):
        """Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz (chest sensor)
        """
        self.sampling_rate = sampling_rate
        self.window_size = int(sampling_rate * self.WINDOW_SIZE_SEC)
        self.window_step = int(self.window_size * (1 - self.WINDOW_OVERLAP))

    def extract_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from time-series data.
        
        Args:
            X: (n_samples, n_features) sensor data
            y: (n_samples,) stress labels
            
        Returns:
            (X_windows, y_windows) where X_windows is (n_windows, window_size*n_features)
        """
        X_windows, y_windows = [], []
        
        for start in range(0, len(X) - self.window_size, self.window_step):
            end = start + self.window_size
            X_windows.append(X[start:end].flatten())
            # Use majority label in window
            y_windows.append(np.bincount(y[start:end]).argmax())
        
        return np.array(X_windows), np.array(y_windows)

    def extract_hrv_features(self, ecg_signal: np.ndarray) -> dict:
        """Extract HRV features from ECG signal.
        
        Time-domain HRV:
        - SDNN: std dev of NN intervals
        - RMSSD: root mean square of successive differences
        - pNN50: % of successive intervals differing >50ms
        
        Frequency-domain HRV:
        - VLF, LF, HF power bands
        - LF/HF ratio (often elevated in stress)
        
        Args:
            ecg_signal: Raw ECG data (assumed to be column 0 of chest sensor)
            
        Returns:
            Dictionary of HRV metrics
        """
        # Simple R-peak detection (in production, use Neurokit2 or similar)
        # High-pass filter to remove baseline wander
        sos = signal.butter(4, 0.5, 'hp', output='sos', fs=self.sampling_rate)
        filtered = signal.sosfilt(sos, ecg_signal)
        
        # Adaptive threshold peak detection
        threshold = np.mean(filtered) + 0.5 * np.std(filtered)
        peaks, _ = signal.find_peaks(filtered, height=threshold, distance=300)  # ~300ms min between peaks
        
        if len(peaks) < 2:
            # Insufficient peaks; return NaN features
            return self._nan_hrv_dict()
        
        # NN intervals (in ms)
        nn_intervals = np.diff(peaks) * 1000 / self.sampling_rate
        
        # Time-domain features
        sdnn = np.std(nn_intervals)
        rmssd = np.sqrt(np.mean(np.diff(nn_intervals) ** 2))
        nn50 = np.sum(np.abs(np.diff(nn_intervals)) > 50)
        pnn50 = 100 * nn50 / (len(nn_intervals) - 1) if len(nn_intervals) > 1 else 0
        
        # Frequency-domain features via Welch PSD
        f, pxx = signal.welch(
            nn_intervals, fs=1000 / np.mean(nn_intervals),
            nperseg=len(nn_intervals) if len(nn_intervals) < 256 else 256
        )
        
        # Power in bands (Hz): VLF [0-0.04], LF [0.04-0.15], HF [0.15-0.4]
        vlf = np.sum(pxx[(f >= 0) & (f < 0.04)])
        lf = np.sum(pxx[(f >= 0.04) & (f < 0.15)])
        hf = np.sum(pxx[(f >= 0.15) & (f < 0.4)])
        lf_hf_ratio = lf / hf if hf > 0 else 0
        
        return {
            'hrv_sdnn': sdnn,
            'hrv_rmssd': rmssd,
            'hrv_pnn50': pnn50,
            'hrv_vlf_power': vlf,
            'hrv_lf_power': lf,
            'hrv_hf_power': hf,
            'hrv_lf_hf_ratio': lf_hf_ratio,
        }

    def extract_respiratory_features(self, resp_signal: np.ndarray) -> dict:
        """Extract respiratory features (breathing rate, depth).
        
        Args:
            resp_signal: Respiration sensor data (column 1 of chest)
            
        Returns:
            Dictionary of respiratory metrics
        """
        # Band-pass filter for respiration (0.1-1 Hz, roughly 6-60 breaths/min)
        sos = signal.butter(4, [0.1, 1], 'bp', output='sos', fs=self.sampling_rate)
        filtered = signal.sosfilt(sos, resp_signal)
        
        # Find breath peaks
        peaks, _ = signal.find_peaks(filtered, distance=100)  # Min 100 samples between peaks
        
        if len(peaks) < 2:
            return self._nan_respiratory_dict()
        
        # Breathing rate (breaths per minute)
        inter_breath_intervals = np.diff(peaks) / self.sampling_rate  # seconds
        breathing_rate = 60 / np.mean(inter_breath_intervals) if len(inter_breath_intervals) > 0 else 0
        
        # Respiratory depth and variability
        breath_amplitude = np.std(filtered[peaks])
        breath_amplitude_cv = np.std(np.diff(filtered[peaks])) / (np.mean(np.abs(filtered[peaks])) + 1e-6)
        
        return {
            'resp_rate': breathing_rate,
            'resp_amplitude': breath_amplitude,
            'resp_variability': breath_amplitude_cv,
        }

    def extract_activity_features(self, wrist_signal: np.ndarray) -> dict:
        """Extract activity features from wrist accelerometer.
        
        Args:
            wrist_signal: Wrist sensor channels (3 accel + gyro + others)
            
        Returns:
            Dictionary of activity metrics
        """
        # Assume first 3 columns are accelerometer (x, y, z)
        if wrist_signal.shape[1] < 3:
            return self._nan_activity_dict()
        
        accel = wrist_signal[:, :3]
        
        # Magnitude of acceleration
        mag = np.linalg.norm(accel, axis=1)
        
        # Activity features
        mean_accel = np.mean(mag)
        std_accel = np.std(mag)
        max_accel = np.max(mag)
        
        # Zero-crossing rate (motion onset detection)
        zcr = np.sum(np.diff(np.sign(accel[:, 0])) != 0) / len(accel)
        
        return {
            'activity_mean': mean_accel,
            'activity_std': std_accel,
            'activity_max': max_accel,
            'activity_zcr': zcr,
        }

    def extract_eda_features(self, eda_signal: np.ndarray) -> dict:
        """Extract electrodermal activity features.
        
        Assumes EDA is one of the wrist channels (typically higher index).
        
        Args:
            eda_signal: EDA/skin conductance data
            
        Returns:
            Dictionary of EDA metrics
        """
        # EDA: high values during stress/arousal
        eda_mean = np.mean(eda_signal)
        eda_std = np.std(eda_signal)
        eda_max = np.max(eda_signal)
        
        # Skin conductance response rate
        eda_diff = np.abs(np.diff(eda_signal))
        scr_rate = np.sum(eda_diff > np.std(eda_diff)) / len(eda_signal)
        
        return {
            'eda_mean': eda_mean,
            'eda_std': eda_std,
            'eda_max': eda_max,
            'eda_response_rate': scr_rate,
        }

    def extract_statistical_features(self, signal_chunk: np.ndarray) -> dict:
        """Extract basic statistical features from any signal chunk.
        
        Args:
            signal_chunk: (window_size, n_features) array
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        for i in range(signal_chunk.shape[1]):
            col = signal_chunk[:, i]
            features[f'stat_ch{i}_mean'] = np.mean(col)
            features[f'stat_ch{i}_std'] = np.std(col)
            features[f'stat_ch{i}_skew'] = skew(col)
            features[f'stat_ch{i}_kurtosis'] = kurtosis(col)
            features[f'stat_ch{i}_min'] = np.min(col)
            features[f'stat_ch{i}_max'] = np.max(col)
        return features

    def extract_all_features(self, X_window: np.ndarray) -> dict:
        """Extract all features from a single window.
        
        Args:
            X_window: Flattened (window_size * n_features,) or reshaped window
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        # Assume structure: chest (3 channels) + wrist (n channels)
        n_chest_ch = 3
        window_size = self.window_size
        
        # Reshape for channel-wise processing
        X_shaped = X_window.reshape(-1, window_size).T  # (window_size, n_channels)
        
        chest_data = X_shaped[:, :n_chest_ch]
        wrist_data = X_shaped[:, n_chest_ch:]
        
        # Extract modality-specific features
        features.update(self.extract_hrv_features(chest_data[:, 0]))      # ECG from chest
        features.update(self.extract_respiratory_features(chest_data[:, 1]))  # Resp
        features.update(self.extract_activity_features(wrist_data))       # Accel/gyro
        features.update(self.extract_eda_features(wrist_data[:, -1]))     # Skin conductance
        features.update(self.extract_statistical_features(X_shaped))      # Statistical features
        
        return features

    def extract_features_from_splits(self, 
                                     train_splits: List[Tuple],
                                     test_splits: List[Tuple]) -> Tuple[List, List]:
        """Extract features for all LOSO splits.
        
        Args:
            train_splits: List of (X_train, y_train, subject_id) tuples
            test_splits: List of (X_test, y_test, subject_id) tuples
            
        Returns:
            (train_features, test_features) as lists of DataFrames
        """
        train_features, test_features = [], []
        
        print("Extracting features...")
        for fold_idx, ((
            X_train, y_train, subject_train), (
            X_test, y_test, subject_test)
        ) in enumerate(zip(train_splits, test_splits)):
            print(f"  Fold {fold_idx} (test on {subject_test})...")
            
            # Create windows
            X_train_win, y_train_win = self.extract_windows(X_train, y_train)
            X_test_win, y_test_win = self.extract_windows(X_test, y_test)
            
            # Extract features per window
            train_feature_list = []
            for x_win in X_train_win:
                feats = self.extract_all_features(x_win)
                train_feature_list.append(feats)
            
            test_feature_list = []
            for x_win in X_test_win:
                feats = self.extract_all_features(x_win)
                test_feature_list.append(feats)
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_feature_list)
            train_df['label'] = y_train_win
            train_df['subject_id'] = subject_train
            train_df['fold'] = fold_idx
            
            test_df = pd.DataFrame(test_feature_list)
            test_df['label'] = y_test_win
            test_df['subject_id'] = subject_test
            test_df['fold'] = fold_idx
            
            train_features.append(train_df)
            test_features.append(test_df)
        
        return train_features, test_features

    # Helper methods
    def _nan_hrv_dict(self) -> dict:
        return {k: np.nan for k in [
            'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
            'hrv_vlf_power', 'hrv_lf_power', 'hrv_hf_power', 'hrv_lf_hf_ratio'
        ]}

    def _nan_respiratory_dict(self) -> dict:
        return {k: np.nan for k in ['resp_rate', 'resp_amplitude', 'resp_variability']}

    def _nan_activity_dict(self) -> dict:
        return {k: np.nan for k in ['activity_mean', 'activity_std', 'activity_max', 'activity_zcr']}
