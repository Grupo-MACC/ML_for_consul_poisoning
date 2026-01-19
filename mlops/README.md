# MLOps Module for Consul Poisoning Detection

This module provides a complete MLOps pipeline for training, tracking, and comparing machine learning models for Consul Poisoning detection.

## Structure

- `config.py` - Configuration settings for the pipeline
- `data_loader.py` - Data loading and preprocessing utilities
- `metrics.py` - Metric calculation functions
- `models.py` - Model creation and training functions
- `train_all_models.py` - Main training script with MLflow tracking
- `compare_models.py` - Model comparison utilities
- `run_mlflow_server.py` - MLflow UI server launcher

## Quick Start

### 1. Install Dependencies

```bash
pip install mlflow scikit-learn hdbscan pandas numpy matplotlib seaborn plotly
```

### 2. Train All Models

```bash
cd mlops
python train_all_models.py
```

### 3. Launch MLflow UI

```bash
python run_mlflow_server.py
```

Then open http://localhost:5000 in your browser.

## Models

1. **GMM (Gaussian Mixture Model)** - Probabilistic clustering
2. **HDBSCAN** - Density-based clustering with outlier detection
3. **Isolation Forest** - Anomaly detection

## Metrics Tracked

### Clustering Metrics
- Silhouette Score
- Homogeneity Score
- Completeness Score
- V-Measure Score

### Anomaly Detection Metrics
- ROC AUC Score
- Average Precision Score
- Precision / Recall / F1 Score
- Confusion Matrix
