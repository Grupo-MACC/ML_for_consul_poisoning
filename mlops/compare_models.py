"""
Model Comparison and Selection Utilities
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME


def get_all_runs(experiment_name=EXPERIMENT_NAME):
    """Get all runs from an experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"]
    )
    
    return runs


def runs_to_dataframe(runs):
    """Convert MLflow runs to a pandas DataFrame."""
    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
        row.update(run.data.params)
        row.update(run.data.metrics)
        data.append(row)
    
    return pd.DataFrame(data)


def compare_models(experiment_name=EXPERIMENT_NAME):
    """
    Compare all models in an experiment and create visualizations.
    """
    runs = get_all_runs(experiment_name)
    df = runs_to_dataframe(runs)
    
    # Filter out data_info run
    df = df[df['run_name'] != 'data_info']
    
    # Extract model type from run_name
    df['model_type'] = df['run_name'].apply(lambda x: x.split('_')[0] if x else 'Unknown')
    
    return df


def plot_model_comparison(df, metric='silhouette_score'):
    """Create a comparison plot for a specific metric across models."""
    if metric not in df.columns:
        available = [c for c in df.columns if not c.startswith('run_') and 
                     c not in ['status', 'start_time', 'end_time', 'model_type']]
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")
    
    fig = px.bar(
        df.sort_values(metric, ascending=False),
        x='run_name',
        y=metric,
        color='model_type',
        title=f'Model Comparison: {metric}',
        labels={'run_name': 'Model Configuration', metric: metric.replace('_', ' ').title()}
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    
    return fig


def get_best_model_per_type(df):
    """Get the best model for each model type."""
    best_models = {}
    
    for model_type in df['model_type'].unique():
        type_df = df[df['model_type'] == model_type]
        
        if model_type == 'IsoForest':
            # Best by ROC AUC
            if 'roc_auc_score' in type_df.columns:
                best = type_df.loc[type_df['roc_auc_score'].idxmax()]
                best_models['IsolationForest'] = best
        else:
            # Best by silhouette score
            if 'silhouette_score' in type_df.columns:
                best = type_df.loc[type_df['silhouette_score'].idxmax()]
                best_models[model_type] = best
    
    return best_models


def create_summary_dashboard(df):
    """Create a comprehensive comparison dashboard."""
    # Define metrics for each model type
    clustering_metrics = ['silhouette_score', 'homogeneity_score', 'completeness_score', 'v_measure_score']
    detection_metrics = ['roc_auc_score', 'average_precision_score', 'precision', 'recall', 'f1_score']
    
    # Filter dataframes
    clustering_df = df[df['model_type'].isin(['GMM', 'HDBSCAN'])]
    detection_df = df[df['model_type'] == 'IsoForest']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Clustering: Silhouette Score',
            'Clustering: V-Measure',
            'Anomaly Detection: ROC AUC',
            'Anomaly Detection: F1 Score'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Clustering - Silhouette
    if 'silhouette_score' in clustering_df.columns:
        sorted_df = clustering_df.sort_values('silhouette_score', ascending=True)
        fig.add_trace(
            go.Bar(x=sorted_df['silhouette_score'], y=sorted_df['run_name'],
                   orientation='h', name='Silhouette', marker_color='steelblue'),
            row=1, col=1
        )
    
    # Clustering - V-Measure
    if 'v_measure_score' in clustering_df.columns:
        sorted_df = clustering_df.sort_values('v_measure_score', ascending=True)
        fig.add_trace(
            go.Bar(x=sorted_df['v_measure_score'], y=sorted_df['run_name'],
                   orientation='h', name='V-Measure', marker_color='forestgreen'),
            row=1, col=2
        )
    
    # Detection - ROC AUC
    if 'roc_auc_score' in detection_df.columns:
        sorted_df = detection_df.sort_values('roc_auc_score', ascending=True)
        fig.add_trace(
            go.Bar(x=sorted_df['roc_auc_score'], y=sorted_df['run_name'],
                   orientation='h', name='ROC AUC', marker_color='coral'),
            row=2, col=1
        )
    
    # Detection - F1
    if 'f1_score' in detection_df.columns:
        sorted_df = detection_df.sort_values('f1_score', ascending=True)
        fig.add_trace(
            go.Bar(x=sorted_df['f1_score'], y=sorted_df['run_name'],
                   orientation='h', name='F1 Score', marker_color='mediumpurple'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=1000,
        width=1200,
        title_text="Model Comparison Dashboard",
        showlegend=False
    )
    
    return fig
