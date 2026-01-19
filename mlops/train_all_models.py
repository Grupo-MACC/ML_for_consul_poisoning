"""
MLflow Training Pipeline for Consul Poisoning Detection Models
Trains all models with hyperparameter search and logs everything to MLflow
"""
import os
import sys
import warnings
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MLFLOW_TRACKING_URI, 
    EXPERIMENT_NAME, 
    MODELS_DIR,
    RANDOM_STATE
)
from data_loader import load_data, get_data_stats
from models import train_gmm, train_hdbscan, train_isolation_forest
from metrics import (
    compute_clustering_metrics, 
    compute_anomaly_detection_metrics,
    compute_cluster_attack_distribution
)

warnings.filterwarnings('ignore')

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def create_confusion_matrix_plot(y_true, y_pred, title, labels=['Normal', 'Attack/Anomaly']):
    """Create and return a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def create_cluster_distribution_plot(labels, y_true, title):
    """Create a bar plot showing attack distribution per cluster."""
    df_temp = pd.DataFrame({'cluster': labels, 'is_attack': y_true})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cluster_counts = df_temp.groupby(['cluster', 'is_attack']).size().unstack(fill_value=0)
    cluster_counts.columns = ['Normal', 'Attack']
    cluster_counts.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'salmon'])
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(title='Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def run_gmm_experiments(X, y, experiment_id):
    """Run GMM experiments with different hyperparameters."""
    print("\n" + "="*60)
    print("🔷 Running GMM Experiments")
    print("="*60)
    
    # Hyperparameter grid
    n_components_list = [2, 3, 4]
    covariance_types = ['full', 'tied', 'diag']
    
    best_run = None
    best_metric = -1
    
    for n_components, cov_type in product(n_components_list, covariance_types):
        run_name = f"GMM_n{n_components}_{cov_type}"
        print(f"\n  📊 Training: {run_name}")
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            start_time = time.time()
            
            # Log parameters
            params = {
                "model_type": "gmm",
                "n_components": n_components,
                "covariance_type": cov_type,
                "reg_covar": 1e-6,
                "preprocessing": "PowerTransformer + StandardScaler"
            }
            mlflow.log_params(params)
            
            # Train model
            pipeline, labels, X_transformed = train_gmm(
                X, n_components=n_components, covariance_type=cov_type
            )
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Compute metrics
            clustering_metrics = compute_clustering_metrics(
                X_transformed, labels, y_true=y.values, exclude_noise=False
            )
            
            # Log clustering metrics
            for metric_name, value in clustering_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)
            
            # Log attack distribution
            distribution = compute_cluster_attack_distribution(labels, y.values)
            for key, value in distribution.items():
                mlflow.log_metric(key, value)
            
            # Create and log confusion matrix (treating cluster with most attacks as "attack" cluster)
            cluster_attack_ratios = {}
            for i in range(n_components):
                mask = labels == i
                if mask.sum() > 0:
                    cluster_attack_ratios[i] = y.values[mask].mean()
            
            attack_cluster = max(cluster_attack_ratios, key=cluster_attack_ratios.get)
            predicted_attacks = (labels == attack_cluster).astype(int)
            
            # Compute anomaly detection metrics
            anomaly_metrics = compute_anomaly_detection_metrics(
                y.values, predicted_attacks
            )
            for metric_name, value in anomaly_metrics.items():
                mlflow.log_metric(f"detection_{metric_name}", value)
            
            # Create and log plots
            fig_cm = create_confusion_matrix_plot(
                y.values, predicted_attacks,
                f"GMM Confusion Matrix (n={n_components}, {cov_type})"
            )
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
            plt.close(fig_cm)
            
            fig_dist = create_cluster_distribution_plot(
                labels, y.values,
                f"GMM Cluster Distribution (n={n_components}, {cov_type})"
            )
            mlflow.log_figure(fig_dist, "cluster_distribution.png")
            plt.close(fig_dist)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Track best model based on silhouette score
            current_metric = clustering_metrics.get("silhouette_score", -1)
            if current_metric > best_metric:
                best_metric = current_metric
                best_run = {
                    "run_id": run.info.run_id,
                    "params": params,
                    "silhouette_score": current_metric,
                    "v_measure": clustering_metrics.get("v_measure_score", 0),
                    "detection_f1": anomaly_metrics.get("f1_score", 0)
                }
            
            print(f"    ✅ Silhouette: {current_metric:.4f} | V-measure: {clustering_metrics.get('v_measure_score', 0):.4f}")
    
    print(f"\n  🏆 Best GMM: n_components={best_run['params']['n_components']}, "
          f"cov_type={best_run['params']['covariance_type']}")
    
    return best_run


def run_hdbscan_experiments(X, y, experiment_id):
    """Run HDBSCAN experiments with different hyperparameters."""
    print("\n" + "="*60)
    print("🔷 Running HDBSCAN Experiments")
    print("="*60)
    
    # Hyperparameter grid (keeping min_cluster_size = min_samples for outlier detection)
    size_params = [50, 80, 100, 120, 150]
    metrics = ['euclidean', 'manhattan']
    
    best_run = None
    best_metric = -1
    
    for size, metric in product(size_params, metrics):
        run_name = f"HDBSCAN_size{size}_{metric}"
        print(f"\n  📊 Training: {run_name}")
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            start_time = time.time()
            
            # Log parameters
            params = {
                "model_type": "hdbscan",
                "min_cluster_size": size,
                "min_samples": size,
                "cluster_selection_method": "eom",
                "metric": metric,
                "preprocessing": "RobustScaler"
            }
            mlflow.log_params(params)
            
            try:
                # Train model
                pipeline, labels, X_transformed = train_hdbscan(
                    X, min_cluster_size=size, min_samples=size, metric=metric
                )
                
                training_time = time.time() - start_time
                mlflow.log_metric("training_time_seconds", training_time)
                
                # Compute metrics
                clustering_metrics = compute_clustering_metrics(
                    X_transformed, labels, y_true=y.values, exclude_noise=True
                )
                
                # Log clustering metrics
                for metric_name, value in clustering_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)
                
                # Log attack distribution
                distribution = compute_cluster_attack_distribution(labels, y.values)
                for key, value in distribution.items():
                    mlflow.log_metric(key, value)
                
                # For HDBSCAN, outliers (label=-1) could be considered anomalies
                predicted_anomalies = (labels == -1).astype(int)
                
                # Compute anomaly detection metrics
                anomaly_metrics = compute_anomaly_detection_metrics(
                    y.values, predicted_anomalies
                )
                for metric_name, value in anomaly_metrics.items():
                    mlflow.log_metric(f"outlier_detection_{metric_name}", value)
                
                # Create and log plots
                fig_cm = create_confusion_matrix_plot(
                    y.values, predicted_anomalies,
                    f"HDBSCAN Outlier Detection (size={size}, {metric})"
                )
                mlflow.log_figure(fig_cm, "confusion_matrix.png")
                plt.close(fig_cm)
                
                fig_dist = create_cluster_distribution_plot(
                    labels, y.values,
                    f"HDBSCAN Cluster Distribution (size={size}, {metric})"
                )
                mlflow.log_figure(fig_dist, "cluster_distribution.png")
                plt.close(fig_dist)
                
                # Log model
                mlflow.sklearn.log_model(pipeline, "model")
                
                # Track best model
                current_metric = clustering_metrics.get("silhouette_score", -1)
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_run = {
                        "run_id": run.info.run_id,
                        "params": params,
                        "silhouette_score": current_metric,
                        "v_measure": clustering_metrics.get("v_measure_score", 0),
                        "noise_ratio": clustering_metrics.get("noise_ratio", 0),
                        "detection_recall": anomaly_metrics.get("recall", 0)
                    }
                
                print(f"    ✅ Silhouette: {current_metric:.4f} | "
                      f"Noise: {clustering_metrics.get('noise_ratio', 0):.2%} | "
                      f"Outlier Recall: {anomaly_metrics.get('recall', 0):.4f}")
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                mlflow.log_param("error", str(e))
                mlflow.set_tag("status", "failed")
    
    if best_run:
        print(f"\n  🏆 Best HDBSCAN: min_size={best_run['params']['min_cluster_size']}, "
              f"metric={best_run['params']['metric']}")
    
    return best_run


def run_isolation_forest_experiments(X, y, experiment_id):
    """Run Isolation Forest experiments with different hyperparameters."""
    print("\n" + "="*60)
    print("🔷 Running Isolation Forest Experiments")
    print("="*60)
    
    # Hyperparameter grid
    n_estimators_list = [50, 100, 150, 200]
    max_samples_list = ['auto', 0.5, 0.8]
    contamination_list = ['auto', 0.01, 0.05]  # 'auto' will use actual attack ratio
    
    best_run = None
    best_metric = -1
    
    for n_est, max_samp, contam in product(n_estimators_list, max_samples_list, contamination_list):
        run_name = f"IsoForest_n{n_est}_samp{max_samp}_cont{contam}"
        print(f"\n  📊 Training: {run_name}")
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            start_time = time.time()
            
            # Calculate actual contamination
            actual_contamination = float(y.mean()) if contam == 'auto' else contam
            
            # Log parameters
            params = {
                "model_type": "isolation_forest",
                "n_estimators": n_est,
                "max_samples": str(max_samp),
                "contamination_param": str(contam),
                "actual_contamination": actual_contamination,
                "preprocessing": "RobustScaler"
            }
            mlflow.log_params(params)
            
            # Train model
            pipeline, labels, scores = train_isolation_forest(
                X, y, n_estimators=n_est, contamination=contam, max_samples=max_samp
            )
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Compute anomaly detection metrics
            anomaly_metrics = compute_anomaly_detection_metrics(
                y.values, labels, anomaly_scores=scores
            )
            
            # Log all metrics
            for metric_name, value in anomaly_metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Additional metrics
            mlflow.log_metric("n_detected_anomalies", int(labels.sum()))
            mlflow.log_metric("detection_ratio", float(labels.mean()))
            
            # Create and log plots
            fig_cm = create_confusion_matrix_plot(
                y.values, labels,
                f"Isolation Forest (n={n_est}, samples={max_samp})"
            )
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
            plt.close(fig_cm)
            
            # Score distribution plot
            fig_score, ax = plt.subplots(figsize=(10, 6))
            df_plot = pd.DataFrame({'score': scores, 'label': y.values})
            for label, group in df_plot.groupby('label'):
                name = 'Attack' if label == 1 else 'Normal'
                ax.hist(group['score'], bins=50, alpha=0.6, label=name)
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Count')
            ax.set_title('Anomaly Score Distribution')
            ax.legend()
            plt.tight_layout()
            mlflow.log_figure(fig_score, "score_distribution.png")
            plt.close(fig_score)
            
            # ROC curve plot
            from sklearn.metrics import RocCurveDisplay
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_predictions(y.values, scores, ax=ax_roc)
            ax_roc.set_title(f'ROC Curve - Isolation Forest (AUC={anomaly_metrics["roc_auc_score"]:.4f})')
            plt.tight_layout()
            mlflow.log_figure(fig_roc, "roc_curve.png")
            plt.close(fig_roc)
            
            # Precision-Recall curve
            from sklearn.metrics import PrecisionRecallDisplay
            fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
            PrecisionRecallDisplay.from_predictions(y.values, scores, ax=ax_pr)
            ax_pr.set_title(f'Precision-Recall Curve (AP={anomaly_metrics["average_precision_score"]:.4f})')
            plt.tight_layout()
            mlflow.log_figure(fig_pr, "precision_recall_curve.png")
            plt.close(fig_pr)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Track best model based on ROC AUC
            current_metric = anomaly_metrics.get("roc_auc_score", -1)
            if current_metric > best_metric:
                best_metric = current_metric
                best_run = {
                    "run_id": run.info.run_id,
                    "params": params,
                    "roc_auc": current_metric,
                    "avg_precision": anomaly_metrics.get("average_precision_score", 0),
                    "f1_score": anomaly_metrics.get("f1_score", 0),
                    "recall": anomaly_metrics.get("recall", 0),
                    "precision": anomaly_metrics.get("precision", 0)
                }
            
            print(f"    ✅ ROC-AUC: {current_metric:.4f} | "
                  f"F1: {anomaly_metrics.get('f1_score', 0):.4f} | "
                  f"Recall: {anomaly_metrics.get('recall', 0):.4f}")
    
    print(f"\n  🏆 Best Isolation Forest: n_estimators={best_run['params']['n_estimators']}, "
          f"max_samples={best_run['params']['max_samples']}")
    
    return best_run


def register_best_models(best_runs, experiment_name):
    """Register the best models in MLflow Model Registry."""
    print("\n" + "="*60)
    print("📦 Registering Best Models")
    print("="*60)
    
    for model_type, run_info in best_runs.items():
        if run_info is None:
            continue
            
        model_name = f"{experiment_name}-{model_type}"
        model_uri = f"runs:/{run_info['run_id']}/model"
        
        try:
            # Register model
            registered_model = mlflow.register_model(model_uri, model_name)
            print(f"  ✅ Registered {model_name} (version {registered_model.version})")
            
            # Add description
            client = mlflow.tracking.MlflowClient()
            client.update_registered_model(
                name=model_name,
                description=f"Best {model_type.upper()} model for Consul Poisoning Detection"
            )
            
            # Tag as production-ready if performance is good
            if model_type == 'isolation_forest' and run_info.get('roc_auc', 0) > 0.9:
                client.set_registered_model_alias(model_name, "production", registered_model.version)
                print(f"    🏷️  Marked as 'production' candidate")
                
        except Exception as e:
            print(f"  ⚠️  Could not register {model_name}: {e}")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("🚀 MLOps Training Pipeline - Consul Poisoning Detection")
    print("="*70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Load data
    print("\n📂 Loading data...")
    X, y, df = load_data()
    data_stats = get_data_stats(X, y)
    
    print(f"   Samples: {data_stats['n_samples']:,}")
    print(f"   Features: {data_stats['n_features']}")
    print(f"   Attacks: {data_stats['n_attacks']:,} ({data_stats['attack_ratio']:.2%})")
    print(f"   Normal: {data_stats['n_normal']:,}")
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "consul-poisoning", "type": "anomaly-detection"}
        )
    else:
        experiment_id = experiment.experiment_id
    
    print(f"\n📋 Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    
    # Log data stats as a parent run
    with mlflow.start_run(experiment_id=experiment_id, run_name="data_info") as parent_run:
        mlflow.log_params({
            "dataset": "windowed_dataset_cleaned.csv",
            "n_samples": data_stats['n_samples'],
            "n_features": data_stats['n_features'],
            "attack_ratio": data_stats['attack_ratio']
        })
        mlflow.log_dict({"feature_names": data_stats['feature_names']}, "feature_names.json")
    
    # Run experiments for each model type
    best_runs = {}
    
    best_runs['gmm'] = run_gmm_experiments(X, y, experiment_id)
    best_runs['hdbscan'] = run_hdbscan_experiments(X, y, experiment_id)
    best_runs['isolation_forest'] = run_isolation_forest_experiments(X, y, experiment_id)
    
    # Register best models
    register_best_models(best_runs, EXPERIMENT_NAME)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    for model_type, run_info in best_runs.items():
        if run_info:
            print(f"\n🔷 {model_type.upper()}")
            print(f"   Run ID: {run_info['run_id']}")
            for key, value in run_info.items():
                if key not in ['run_id', 'params']:
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print(f"✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 View results at: http://localhost:5000")
    print("="*70)


if __name__ == "__main__":
    main()
