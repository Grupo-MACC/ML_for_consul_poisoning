"""
MLOps Configuration for Consul Poisoning Detection
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlops" / "mlruns"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "preprocessed_datasets" / "windowed_dataset_cleaned.csv"

# MLflow configuration
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR.absolute()}"
EXPERIMENT_NAME = "consul-poisoning-detection"

# Model configurations
MODELS_CONFIG = {
    "gmm": {
        "name": "Gaussian Mixture Model",
        "params": {
            "n_components": [2, 3, 4, 5],
            "covariance_type": ["full", "tied", "diag"],
            "reg_covar": 1e-6,
            "random_state": 42
        },
        "preprocessing": {
            "power_transform": True,
            "standard_scaler": True
        }
    },
    "hdbscan": {
        "name": "HDBSCAN Clustering",
        "params": {
            "min_cluster_size": [50, 80, 100, 120, 150],
            "min_samples": [50, 80, 100, 120, 150],
            "cluster_selection_method": "eom",
            "metric": "euclidean"
        },
        "preprocessing": {
            "robust_scaler": True
        }
    },
    "isolation_forest": {
        "name": "Isolation Forest",
        "params": {
            "n_estimators": [50, 100, 200],
            "contamination": "auto",  # Will be calculated from data
            "max_samples": ["auto", 0.5, 0.8],
            "random_state": 42,
            "n_jobs": -1
        },
        "preprocessing": {
            "robust_scaler": True
        }
    }
}

# Evaluation metrics to track
CLUSTERING_METRICS = [
    "silhouette_score",
    "homogeneity_score", 
    "completeness_score",
    "v_measure_score"
]

ANOMALY_DETECTION_METRICS = [
    "roc_auc_score",
    "average_precision_score",
    "precision",
    "recall",
    "f1_score"
]

# Random seed for reproducibility
RANDOM_STATE = 42
