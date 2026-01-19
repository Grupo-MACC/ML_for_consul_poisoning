"""
Data loading utilities for MLOps pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import TRAIN_DATA_PATH, RANDOM_STATE


def load_data(path=None, sample_size=None):
    """
    Load and prepare the dataset for training.
    
    Args:
        path: Path to the CSV file. Uses default if None.
        sample_size: Number of samples to use. Uses all if None.
    
    Returns:
        X: Features DataFrame
        y: Target Series
        df: Full DataFrame
    """
    data_path = path or TRAIN_DATA_PATH
    df = pd.read_csv(data_path)
    
    if sample_size and sample_size < len(df):
        df, _ = train_test_split(
            df, 
            train_size=sample_size, 
            stratify=df['is_attack'],
            random_state=RANDOM_STATE
        )
        df = df.reset_index(drop=True)
    
    X = df[df.columns.difference(['is_attack'])]
    y = df['is_attack']
    
    return X, y, df


def get_data_stats(X, y):
    """
    Get basic statistics about the dataset.
    
    Args:
        X: Features DataFrame
        y: Target Series
    
    Returns:
        dict: Dictionary with data statistics
    """
    stats = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_attacks": int(y.sum()),
        "n_normal": int((~y.astype(bool)).sum()),
        "attack_ratio": float(y.mean()),
        "feature_names": list(X.columns)
    }
    return stats


def compute_mahalanobis_vi(X):
    """
    Compute the inverse covariance matrix for Mahalanobis distance.
    
    Args:
        X: Features DataFrame or array
    
    Returns:
        VI: Inverse covariance matrix
    """
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    cov = np.cov(X_scaled, rowvar=False)
    
    # Add regularization for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6
    VI = np.linalg.inv(cov)
    
    return VI
