"""
Módulo para visualización de clustering
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_samples, 
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)


def plot_silhouette(ax, X, labels, title, max_height=800):
    """
    Dibuja silhouette excluyendo outliers y reescalando clusters grandes
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde dibujar
    X : array-like
        Datos originales
    labels : array-like
        Etiquetas de clustering (-1 para outliers)
    title : str
        Título del gráfico
    max_height : int, default=800
        Altura máxima para submuestreo visual
        
    Returns
    -------
    float or None
        Silhouette score promedio, o None si hay menos de 2 clusters
    """
    mask = labels != -1
    Xc = X[mask]
    lc = labels[mask]

    # Necesitamos al menos 2 clusters
    if len(np.unique(lc)) < 2:
        ax.set_title(f"{title}\n<2 clusters")
        ax.axis("off")
        return None

    sil_vals = silhouette_samples(Xc, lc)
    sil_avg = silhouette_score(Xc, lc)

    y_lower = 10

    for cluster in np.unique(lc):
        vals = sil_vals[lc == cluster]
        vals.sort()

        # Subsample visual
        if len(vals) > max_height:
            idx = np.linspace(0, len(vals) - 1, max_height).astype(int)
            vals_plot = vals[idx]
        else:
            vals_plot = vals

        size = len(vals_plot)
        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            vals_plot,
            alpha=0.7
        )

        y_lower = y_upper + 10

    ax.axvline(sil_avg, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(-0.2, 1.0)
    ax.set_yticks([])
    ax.set_title(f"{title}\nSil={sil_avg:.3f}")

    return sil_avg


def compare_hdbscan_params(X, param_grid, y_true=None, metric="euclidean", 
                          cluster_selection_method="eom", 
                          n_cols=2, figsize_per_row=4):
    """
    Compara diferentes configuraciones de HDBSCAN usando silhouette plots
    
    Parameters
    ----------
    X : array-like
        Datos a clusterizar
    param_grid : list of tuples
        Lista de (min_cluster_size, min_samples)
    y_true : array-like, optional
        Etiquetas verdaderas para calcular métricas supervisadas
    metric : str, default="euclidean"
        Métrica de distancia
    cluster_selection_method : str, default="eom"
        Método de selección de clusters
    n_cols : int, default=2
        Número de columnas en la grilla
    figsize_per_row : int, default=4
        Altura por fila en la figura
        
    Returns
    -------
    results : list of dict
        Métricas para cada configuración
    fig : matplotlib.figure.Figure
        Figura generada
    """
    import hdbscan
    
    n_rows = int(np.ceil(len(param_grid) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, figsize_per_row * n_rows),
        sharex=True
    )

    axes = axes.flatten()

    results = []

    for ax, (mcs, ms) in zip(axes, param_grid):
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method=cluster_selection_method,
            metric=metric
        )

        labels = hdb.fit_predict(X)

        sil = plot_silhouette(
            ax,
            X,
            labels,
            title=f"min_cluster_size={mcs}\nmin_samples={ms}"
        )

        result = {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "silhouette": sil,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_outliers": np.sum(labels == -1)
        }
        
        # Si tenemos etiquetas verdaderas, calcular métricas supervisadas
        if y_true is not None:
            # Excluir outliers para métricas supervisadas
            mask = labels != -1
            if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
                result["homogeneity"] = homogeneity_score(y_true[mask], labels[mask])
                result["completeness"] = completeness_score(y_true[mask], labels[mask])
                result["v_measure"] = v_measure_score(y_true[mask], labels[mask])
            else:
                result["homogeneity"] = None
                result["completeness"] = None
                result["v_measure"] = None
        
        results.append(result)

    # Apagar ejes sobrantes
    for ax in axes[len(param_grid):]:
        ax.axis("off")

    plt.suptitle("Comparación de Silhouette – HDBSCAN (sin outliers)", fontsize=14)
    plt.tight_layout()

    return results, fig


import matplotlib.pyplot as plt

def plot_metrics_comparison(results, metrics=None, figsize=(14, 5)):
    """
    Visualiza comparación de métricas para diferentes configuraciones
    
    Parameters
    ----------
    results : list of dict
        Resultados de compare_hdbscan_params
    metrics : list of str, optional
        Métricas a visualizar. Por defecto usa todas las disponibles.
    figsize : tuple, default=(14, 5)
        Tamaño de la figura
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Crear etiquetas limpias para el eje x (sin .0)
    df['config'] = df.apply(
        lambda row: f"({int(row['min_cluster_size'])}, {int(row['min_samples'])})", 
        axis=1
    )
    
    # Determinar qué métricas visualizar
    available_metrics = []
    if 'silhouette' in df.columns and df['silhouette'].notna().any():
        available_metrics.append('silhouette')
    if 'homogeneity' in df.columns and df['homogeneity'].notna().any():
        available_metrics.extend(['homogeneity', 'completeness', 'v_measure'])
    
    if metrics is None:
        metrics = available_metrics
    else:
        # Filtrar solo las métricas que están disponibles
        metrics = [m for m in metrics if m in available_metrics]
    
    if not metrics:
        print("No hay métricas disponibles para visualizar")
        return None
    
    # Ajustar figsize dinámicamente si hay muchas configuraciones
    n_configs = len(df['config'].unique())
    if n_configs > 6:
        figsize = (min(18, n_configs * 1.5), 5)
    
    # Crear figura
    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = {
        'silhouette': '#27ae60',   # verde oscuro
        'homogeneity': '#2980b9', # azul oscuro
        'completeness': '#e74c3c', # rojo vivo
        'v_measure': '#8e44ad'     # morado
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filtrar valores no nulos
        mask = df[metric].notna()
        x_vals = df[mask]['config']
        y_vals = df[mask][metric]
        
        bars = ax.bar(x_vals, y_vals, color=colors.get(metric, '#95a5a6'), alpha=0.85, edgecolor='black', linewidth=1)
        ax.set_xlabel('(min_cluster_size, min_samples)', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotar etiquetas si hay muchas
        if len(x_vals) > 4:
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        else:
            ax.tick_params(axis='x', labelsize=10)
        
        # Añadir valores sobre las barras
        for j, (x, y) in enumerate(zip(x_vals, y_vals)):
            ax.text(j, y + 0.01, f'{y:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color='black', alpha=0.8)
    
    # Ocultar ejes sobrantes
    for ax in axes[len(metrics):]:
        ax.axis('off')
    
    plt.suptitle('Comparación de Métricas por Configuración', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # espacio para el suptitle
    
    return fig