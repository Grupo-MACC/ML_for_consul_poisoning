"""
Model training utilities for MLOps pipeline
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import hdbscan

from config import RANDOM_STATE


def create_gmm_pipeline(n_components=2, covariance_type='full', reg_covar=1e-6):
    """
    Create a GMM pipeline with preprocessing.
    
    Args:
        n_components: Number of Gaussian components
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        reg_covar: Regularization for covariance
    
    Returns:
        Pipeline: Scikit-learn pipeline with GMM
    """
    pipeline = Pipeline([
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler', StandardScaler()),
        ('gmm', GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=RANDOM_STATE,
            max_iter=200,
            n_init=5
        ))
    ])
    return pipeline


def create_hdbscan_pipeline(min_cluster_size=120, min_samples=120, 
                            cluster_selection_method='eom', metric='euclidean'):
    """
    Create an HDBSCAN pipeline with preprocessing.
    
    Args:
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        cluster_selection_method: 'eom' or 'leaf'
        metric: Distance metric
    
    Returns:
        Pipeline: Scikit-learn pipeline with HDBSCAN
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('hdbscan', hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
            core_dist_n_jobs=-1
        ))
    ])
    return pipeline


def create_isolation_forest_pipeline(n_estimators=100, contamination='auto',
                                     max_samples='auto'):
    """
    Create an Isolation Forest pipeline with preprocessing.
    
    Args:
        n_estimators: Number of trees
        contamination: Proportion of outliers or 'auto'
        max_samples: Samples to draw for training each tree
    
    Returns:
        Pipeline: Scikit-learn pipeline with Isolation Forest
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('isolation_forest', IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    return pipeline


def train_gmm(X, n_components=2, covariance_type='full', reg_covar=1e-6):
    """
    Train a GMM model and return predictions.
    
    Args:
        X: Feature matrix
        n_components: Number of components
        covariance_type: Covariance type
        reg_covar: Regularization
    
    Returns:
        pipeline: Trained pipeline
        labels: Cluster labels
        X_transformed: Transformed features
    """
    pipeline = create_gmm_pipeline(n_components, covariance_type, reg_covar)
    labels = pipeline.fit_predict(X)
    
    # Get transformed features for metric calculation
    X_transformed = pipeline.named_steps['power_transform'].transform(X)
    X_transformed = pipeline.named_steps['scaler'].transform(X_transformed)
    
    return pipeline, labels, X_transformed


def train_hdbscan(X, min_cluster_size=120, min_samples=120,
                  cluster_selection_method='eom', metric='euclidean'):
    """
    Train an HDBSCAN model and return predictions.
    
    Args:
        X: Feature matrix
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples
        cluster_selection_method: Selection method
        metric: Distance metric
    
    Returns:
        pipeline: Trained pipeline
        labels: Cluster labels
        X_transformed: Transformed features
    """
    pipeline = create_hdbscan_pipeline(
        min_cluster_size, min_samples, cluster_selection_method, metric
    )
    labels = pipeline.fit_predict(X)
    
    # Get transformed features
    X_transformed = pipeline.named_steps['scaler'].transform(X)
    
    return pipeline, labels, X_transformed


def train_isolation_forest(X, y=None, n_estimators=100, contamination='auto',
                           max_samples='auto'):
    """
    Train an Isolation Forest model and return predictions.
    
    Args:
        X: Feature matrix
        y: True labels (used to calculate contamination if needed)
        n_estimators: Number of trees
        contamination: Proportion of outliers
        max_samples: Samples for training
    
    Returns:
        pipeline: Trained pipeline
        labels: Anomaly labels (1=anomaly, 0=normal)
        scores: Anomaly scores
    """
    # Calculate contamination from data if needed
    if contamination == 'auto' and y is not None:
        contamination = float(y.mean())
    elif contamination == 'auto':
        contamination = 0.1  # Default
    
    pipeline = create_isolation_forest_pipeline(
        n_estimators, contamination, max_samples
    )
    pipeline.fit(X)
    
    # Get predictions and scores
    raw_predictions = pipeline.predict(X)
    labels = (raw_predictions == -1).astype(int)  # 1=anomaly, 0=normal
    scores = -pipeline.decision_function(X)  # Higher = more anomalous
    
    return pipeline, labels, scores
