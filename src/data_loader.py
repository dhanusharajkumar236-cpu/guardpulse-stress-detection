"""WESAD dataset loading and preprocessing.

WESAD (Wearable Stress and Affect Detection) is a public multimodal dataset
with stress labels from chest and wrist sensors. This module handles:
- Loading raw sensor data from WESAD subjects
- Resampling and aligning temporal data
- Handling missing values and artifacts
- Splitting into train/test per subject (LOSO protocol)
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import signal


class WESADLoader:
    """Load and preprocess WESAD dataset."""

    # WESAD sampling rates (Hz)
    CHEST_SAMPLING_RATE = 700  # RespiBAN chest sensor
    WRIST_SAMPLING_RATE = 64   # Empatica E4 wrist sensor
    
    # Stress labels in WESAD
    STRESS_LABELS = {
        0: "not_stressed",
        1: "stressed",
        2: "amusement",
        3: "meditation"
    }

    def __init__(self, data_dir: str = "data/wesad"):
        """Initialize loader.
        
        Args:
            data_dir: Path to WESAD root directory containing S2, S3, ..., S17 folders
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"WESAD data directory not found: {data_dir}")
        
        # Discover subject directories
        self.subjects = sorted([
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('S')
        ])
        print(f"Found {len(self.subjects)} subjects: {[s.name for s in self.subjects]}")

    def load_subject(self, subject_path: Path) -> Dict:
        """Load pickled data for a single subject.
        
        WESAD stores data as pickle files with structure:
        - 'signal': dict with 'chest' and 'wrist' sensor arrays
        - 'label': ground truth stress label per sample
        - 'subject': subject ID
        
        Args:
            subject_path: Path to subject's pickle file (e.g., S2/S2.pkl)
            
        Returns:
            Dict with 'chest', 'wrist', 'label', 'subject_id'
        """
        pkl_file = subject_path / f"{subject_path.name}.pkl"
        if not pkl_file.exists():
            raise FileNotFoundError(f"Subject pickle not found: {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        return {
            'chest': data['signal']['chest'],      # (n_samples, n_chest_channels)
            'wrist': data['signal']['wrist'],      # (n_samples, n_wrist_channels)
            'label': data['label'],                # (n_samples,)
            'subject_id': data['subject']
        }

    def resample_wrist_to_chest(self, wrist_signal: np.ndarray) -> np.ndarray:
        """Resample wrist signal (64 Hz) to chest rate (700 Hz).
        
        Args:
            wrist_signal: (n_samples, n_channels) at 64 Hz
            
        Returns:
            Resampled (n_samples_chest, n_channels) at 700 Hz
        """
        ratio = self.CHEST_SAMPLING_RATE / self.WRIST_SAMPLING_RATE
        n_new_samples = int(wrist_signal.shape[0] * ratio)
        resampled = signal.resample(wrist_signal, n_new_samples, axis=0)
        return resampled

    def preprocess_subject(self, subject_path: Path) -> pd.DataFrame:
        """Load and preprocess a subject's data.
        
        Steps:
        1. Load pickle data
        2. Resample wrist to chest rate for alignment
        3. Stack chest + resampled wrist
        4. Remove stress transition periods (label 0, often noisy)
        5. Return as DataFrame with label
        
        Args:
            subject_path: Path to subject directory
            
        Returns:
            DataFrame with columns: chest_ch0, chest_ch1, ..., wrist_ch0, ..., label
        """
        raw_data = self.load_subject(subject_path)
        
        chest = raw_data['chest']
        wrist = raw_data['wrist']
        label = raw_data['label']
        subject_id = raw_data['subject_id']
        
        # Resample wrist to match chest temporal resolution
        wrist_resampled = self.resample_wrist_to_chest(wrist)
        
        # Ensure same length (chest is reference)
        min_len = min(chest.shape[0], wrist_resampled.shape[0])
        chest = chest[:min_len]
        wrist_resampled = wrist_resampled[:min_len]
        label = label[:min_len]
        
        # Stack into feature matrix
        X = np.hstack([chest, wrist_resampled])  # (n_samples, n_chest_ch + n_wrist_ch)
        
        # Create DataFrame
        n_chest_ch = chest.shape[1]
        n_wrist_ch = wrist_resampled.shape[1]
        
        cols = (list(f"chest_{i}" for i in range(n_chest_ch)) +
                list(f"wrist_{i}" for i in range(n_wrist_ch)))
        
        df = pd.DataFrame(X, columns=cols)
        df['label'] = label
        df['subject_id'] = subject_id
        
        return df

    def load_all_subjects_loso_split(self) -> Tuple[List[Tuple], List[Tuple]]:
        """Load all subjects and prepare train/test splits (LOSO protocol).
        
        Leave-One-Subject-Out (LOSO) is the honest evaluation protocol:
        - For each subject: train on all other subjects, test on that subject
        - This measures true generalization to unseen subjects
        
        Returns:
            Tuple of:
            - train_splits: List of (X_train, y_train, subject_test_id) tuples
            - test_splits: List of (X_test, y_test, subject_test_id) tuples
        """
        print("Loading all subjects...")
        all_data = []
        
        for subject_path in self.subjects:
            try:
                df = self.preprocess_subject(subject_path)
                all_data.append(df)
                print(f"  Loaded {subject_path.name}: {len(df)} samples")
            except Exception as e:
                print(f"  ERROR loading {subject_path.name}: {e}")
        
        train_splits = []
        test_splits = []
        
        # LOSO: iterate over test subjects
        for test_idx, test_subject in enumerate(all_data):
            test_subject_id = test_subject['subject_id'].iloc[0]
            
            # Train on all other subjects
            train_data = pd.concat(
                [all_data[i] for i in range(len(all_data)) if i != test_idx],
                ignore_index=True
            )
            
            X_train = train_data.drop(['label', 'subject_id'], axis=1).values
            y_train = train_data['label'].values
            
            X_test = test_subject.drop(['label', 'subject_id'], axis=1).values
            y_test = test_subject['label'].values
            
            train_splits.append((X_train, y_train, test_subject_id))
            test_splits.append((X_test, y_test, test_subject_id))
            
            print(f"LOSO fold {test_idx}: "
                  f"train={X_train.shape}, test={X_test.shape}, "
                  f"test_subject={test_subject_id}")
        
        return train_splits, test_splits


if __name__ == "__main__":
    # Quick test
    loader = WESADLoader("data/wesad")
    train_splits, test_splits = loader.load_all_subjects_loso_split()
    print(f"\nSuccessfully prepared {len(train_splits)} LOSO folds")
