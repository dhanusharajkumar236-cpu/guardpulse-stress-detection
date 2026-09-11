"""LOSO cross-validation and evaluation framework.

Leave-One-Subject-Out (LOSO) evaluation is the honest protocol for stress detection:
- Train on all subjects except one
- Test on the held-out subject
- Repeat for each subject
- Reports true generalization to unseen individuals

This prevents overstating accuracy from evaluating on the same subjects used for training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import json
from pathlib import Path
import joblib


class LOSOEvaluator:
    """Leave-One-Subject-Out cross-validation evaluator."""
    
    def __init__(self, model_class, output_dir: str = "results"):
        """Initialize evaluator.
        
        Args:
            model_class: Model class to instantiate (e.g., StressDetectionModel)
            output_dir: Directory to save results
        """
        self.model_class = model_class
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fold_results = []
        self.models = []  # Store trained models per fold
    
    def evaluate_loso(self, train_features: List[pd.DataFrame], 
                      test_features: List[pd.DataFrame]) -> Dict:
        """Run LOSO evaluation across all folds.
        
        Args:
            train_features: List of training feature DataFrames (one per fold)
            test_features: List of test feature DataFrames (one per fold)
            
        Returns:
            Dictionary with aggregated metrics and per-fold results
        """
        print("Starting LOSO cross-validation...\\n")
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold_idx, (train_df, test_df) in enumerate(zip(train_features, test_features)):
            test_subject = test_df['subject_id'].iloc[0]
            print(f"Fold {fold_idx}: Test subject = {test_subject}")
            
            # Extract features and labels
            feature_cols = [c for c in train_df.columns if c not in ['label', 'subject_id', 'fold']]
            
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df['label'].values
            
            X_test = test_df[feature_cols].fillna(0).values
            y_test = test_df['label'].values
            
            # Initialize and train model
            model = self.model_class(model_type='rf')
            model.fit(X_train, y_train)
            self.models.append(model)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate fold
            fold_metrics = model.evaluate(y_test, y_pred)
            fold_metrics['test_subject'] = test_subject
            fold_metrics['n_test_samples'] = len(y_test)
            
            self.fold_results.append(fold_metrics)
            
            print(f"  Accuracy: {fold_metrics['accuracy']:.3f}, "
                  f"F1: {fold_metrics['f1']:.3f}")
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(np.max(y_proba, axis=1))  # Max confidence per sample
        
        print("\\n" + "="*50)
        print("LOSO RESULTS (Aggregated)")
        print("="*50)
        
        # Aggregate metrics
        aggregate = {
            'accuracy': accuracy_score(all_y_true, all_y_pred),
            'precision': precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            'recall': recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            'f1': f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            'n_folds': len(self.fold_results),
            'n_samples': len(all_y_true),
        }
        
        # Per-fold statistics
        fold_accs = [r['accuracy'] for r in self.fold_results]
        fold_f1s = [r['f1'] for r in self.fold_results]
        
        aggregate['fold_accuracy_mean'] = np.mean(fold_accs)
        aggregate['fold_accuracy_std'] = np.std(fold_accs)
        aggregate['fold_f1_mean'] = np.mean(fold_f1s)
        aggregate['fold_f1_std'] = np.std(fold_f1s)
        
        print(f"Overall Accuracy: {aggregate['accuracy']:.3f}")
        print(f"  Mean ± Std across folds: {aggregate['fold_accuracy_mean']:.3f} ± {aggregate['fold_accuracy_std']:.3f}")
        print(f"Overall F1: {aggregate['f1']:.3f}")
        print(f"  Mean ± Std across folds: {aggregate['fold_f1_mean']:.3f} ± {aggregate['fold_f1_std']:.3f}")
        print(f"\\nTotal samples: {aggregate['n_samples']} across {aggregate['n_folds']} folds")
        
        return {
            'aggregate': aggregate,
            'fold_results': self.fold_results,
            'all_y_true': all_y_true,
            'all_y_pred': all_y_pred,
            'all_y_proba': all_y_proba,
        }
    
    def save_results(self, results: Dict, filename: str = "loso_results.json"):
        """Save evaluation results to JSON.
        
        Args:
            results: Results dictionary from evaluate_loso()
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            'aggregate': results['aggregate'],
            'fold_results': results['fold_results'],
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\\nResults saved to {output_path}")
    
    def save_models(self, base_path: str = "models"):
        """Save trained models for each fold.
        
        Args:
            base_path: Directory to save models
        """
        model_dir = Path(base_path)
        model_dir.mkdir(exist_ok=True)
        
        for fold_idx, model in enumerate(self.models):
            model_path = model_dir / f"fold_{fold_idx}_model.joblib"
            joblib.dump(model, model_path)
        
        print(f"Saved {len(self.models)} trained models to {model_dir}")
