"""
Metrics calculation utilities for MLOps pipeline
"""
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)


def compute_clustering_metrics(X_transformed, labels, y_true=None, exclude_noise=True):
    """
    Compute clustering evaluation metrics.
    
    Args:
        X_transformed: Transformed feature matrix
        labels: Cluster labels
        y_true: Ground truth labels (optional, for semi-supervised metrics)
        exclude_noise: Whether to exclude noise points (label=-1)
    
    Returns:
        dict: Dictionary with metric names and values
    """
    metrics = {}
    
    # Filter out noise points if needed
    if exclude_noise:
        mask = labels != -1
        X_eval = X_transformed[mask]
        labels_eval = labels[mask]
        y_eval = y_true[mask] if y_true is not None else None
        metrics["noise_ratio"] = 1 - mask.mean()
        metrics["n_noise_points"] = int((~mask).sum())
    else:
        X_eval = X_transformed
        labels_eval = labels
        y_eval = y_true
        metrics["noise_ratio"] = 0.0
        metrics["n_noise_points"] = 0
    
    # Number of clusters
    unique_labels = np.unique(labels_eval)
    metrics["n_clusters"] = len(unique_labels)
    
    # Internal metrics (don't need ground truth)
    if len(unique_labels) > 1:
        metrics["silhouette_score"] = float(silhouette_score(X_eval, labels_eval))
        
        # Silhouette per cluster
        sil_samples = silhouette_samples(X_eval, labels_eval)
        for label in unique_labels:
            cluster_mask = labels_eval == label
            metrics[f"silhouette_cluster_{label}"] = float(sil_samples[cluster_mask].mean())
            metrics[f"cluster_{label}_size"] = int(cluster_mask.sum())
    else:
        metrics["silhouette_score"] = 0.0
    
    # External metrics (need ground truth)
    if y_eval is not None and len(unique_labels) > 1:
        metrics["homogeneity_score"] = float(homogeneity_score(y_eval, labels_eval))
        metrics["completeness_score"] = float(completeness_score(y_eval, labels_eval))
        metrics["v_measure_score"] = float(v_measure_score(y_eval, labels_eval))
    
    return metrics


def compute_anomaly_detection_metrics(y_true, y_pred, anomaly_scores=None):
    """
    Compute anomaly detection evaluation metrics.
    
    Args:
        y_true: Ground truth labels (1=anomaly, 0=normal)
        y_pred: Predicted labels (1=anomaly, 0=normal)
        anomaly_scores: Continuous anomaly scores (optional)
    
    Returns:
        dict: Dictionary with metric names and values
    """
    metrics = {}
    
    # Classification metrics
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    
    # Derived metrics
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["detection_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # Score-based metrics (if scores provided)
    if anomaly_scores is not None:
        metrics["roc_auc_score"] = float(roc_auc_score(y_true, anomaly_scores))
        metrics["average_precision_score"] = float(average_precision_score(y_true, anomaly_scores))
        
        # Precision-Recall AUC
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, anomaly_scores)
        metrics["pr_auc"] = float(auc(recall_arr, precision_arr))
    
    return metrics


def compute_cluster_attack_distribution(labels, y_true):
    """
    Compute the distribution of attacks across clusters.
    
    Args:
        labels: Cluster labels
        y_true: Ground truth attack labels
    
    Returns:
        dict: Dictionary with attack distribution per cluster
    """
    distribution = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_name = "noise" if label == -1 else f"cluster_{label}"
        
        n_total = int(cluster_mask.sum())
        n_attacks = int(y_true[cluster_mask].sum())
        n_normal = n_total - n_attacks
        attack_ratio = n_attacks / n_total if n_total > 0 else 0.0
        
        distribution[f"{cluster_name}_total"] = n_total
        distribution[f"{cluster_name}_attacks"] = n_attacks
        distribution[f"{cluster_name}_normal"] = n_normal
        distribution[f"{cluster_name}_attack_ratio"] = float(attack_ratio)
    
    return distribution
