"""Scikit-learn baseline models for stress detection."""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class StressDetectionModel:
    """Scikit-learn model for stress classification."""
    
    def __init__(self, model_type='rf'):
        """Initialize model.
        
        Args:
            model_type: 'rf' (Random Forest) or 'gb' (Gradient Boosting)
        """
        if model_type == 'rf':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])
    
    def fit(self, X, y):
        """Fit model."""
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict labels."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities (confidence)."""
        return self.pipeline.named_steps['model'].predict_proba(X)
    
    @staticmethod
    def evaluate(y_true, y_pred):
        """Evaluate predictions."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
