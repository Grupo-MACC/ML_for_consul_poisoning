"""
Módulo para visualización de clustering con HDBSCAN
(distancia de Mahalanobis soportada correctamente)
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


# ============================================================
# Utilidades de distancia
# ============================================================

def compute_mahalanobis_VI(X):
    """
    Calcula la inversa de la matriz de covarianza (VI)
    necesaria para la distancia de Mahalanobis.
    """
    cov_matrix = np.cov(X, rowvar=False)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # regularización

    try:
        VI = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        VI = np.linalg.pinv(cov_matrix)

    return VI


def compute_mahalanobis_distance_matrix(X, VI):
    """
    Calcula la matriz de distancias de Mahalanobis
    """
    from scipy.spatial.distance import pdist, squareform
    dist_condensed = pdist(X, metric="mahalanobis", VI=VI)
    return squareform(dist_condensed)


# ============================================================
# Silhouette plot
# ============================================================
def plot_silhouette_eucl(ax, X, labels, title, max_height=800):
    """
    Dibuja silhouette excluyendo outliers y reescalando clusters grandes
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde dibujar
    X : array-like
        Datos escalados (deben estar en la misma escala que usó el clustering)
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



def plot_silhouette(ax, X, labels, title, VI, max_height=800):
    """
    Dibuja silhouette excluyendo outliers (-1)
    
    Parameters
    ----------
    X : array-like
        Datos escalados (deben estar en la misma escala que usó el clustering)
    """

    mask = labels != -1
    Xc = X[mask]
    lc = labels[mask]

    if len(np.unique(lc)) < 2:
        ax.set_title(f"{title}\n<2 clusters")
        ax.axis("off")
        return None

    sil_vals = silhouette_samples(
        Xc,
        lc,
        metric="mahalanobis",
        VI=VI
    )

    sil_avg = silhouette_score(
        Xc,
        lc,
        metric="mahalanobis",
        VI=VI
    )

    y_lower = 10

    for cluster in np.unique(lc):
        vals = sil_vals[lc == cluster]
        vals.sort()

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


# ============================================================
# Comparación de parámetros HDBSCAN
# ============================================================
def compare_hdbscan_params_fast(
    X,
    param_grid,
    y_true=None,
    metric="mahalanobis",
    cluster_selection_method="eom",
    n_cols=2,
    figsize_per_row=4
):
    """
    Versión optimizada que NO precalcula la matriz de distancias
    
    IMPORTANTE: Ahora escala los datos ANTES de calcular VI y antes de clustering
    """
    import hdbscan
    from sklearn.preprocessing import RobustScaler

    if metric != "mahalanobis":
        raise ValueError("Este módulo está diseñado para Mahalanobis")

    # CORRECCIÓN: Escalar datos ANTES de todo
    print("Escalando datos...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Calculando VI para Mahalanobis sobre datos escalados...")
    VI = compute_mahalanobis_VI(X_scaled)

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
        print(f"Procesando min_cluster_size={mcs}, min_samples={ms}...")
        
        # CORRECCIÓN: Usar HDBSCAN directamente sin pipeline
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method=cluster_selection_method,
            metric="mahalanobis",
            VI=VI,
            core_dist_n_jobs=-1
        )
        
        # CORRECCIÓN: Usar datos escalados
        labels = hdb.fit_predict(X_scaled)

        # CORRECCIÓN: Calcular silhouette sobre datos escalados
        sil = plot_silhouette(
            ax=ax,
            X=X_scaled,  # Usar datos escalados
            labels=labels,
            title=f"min_cluster_size={mcs}\nmin_samples={ms}",
            VI=VI
        )

        result = {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "silhouette": sil,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_outliers": int(np.sum(labels == -1))
        }

        if y_true is not None:
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

    for ax in axes[len(param_grid):]:
        ax.axis("off")

    plt.suptitle(
        "Comparación de Silhouette – HDBSCAN (Mahalanobis, sin outliers)",
        fontsize=14
    )

    plt.tight_layout()
    return results, fig

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
    from sklearn.preprocessing import RobustScaler
    
    # CORRECCIÓN: Escalar datos ANTES de todo
    print("Escalando datos...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
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
        # CORRECCIÓN: Usar HDBSCAN directamente sin pipeline
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method=cluster_selection_method,
            metric=metric
        )

        # CORRECCIÓN: Usar datos escalados
        labels = hdb.fit_predict(X_scaled)

        # CORRECCIÓN: Calcular silhouette sobre datos escalados
        sil = plot_silhouette_eucl(
            ax,
            X_scaled,  # Usar datos escalados
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

# ============================================================
# Visualización de métricas
# ============================================================

def plot_metrics_comparison(results, metrics=None, figsize=(14, 5)):
    import pandas as pd

    df = pd.DataFrame(results)

    df["config"] = df.apply(
        lambda r: f"({int(r['min_cluster_size'])}, {int(r['min_samples'])})",
        axis=1
    )

    available = []
    if df["silhouette"].notna().any():
        available.append("silhouette")
    if "homogeneity" in df.columns and df["homogeneity"].notna().any():
        available += ["homogeneity", "completeness", "v_measure"]

    if metrics is None:
        metrics = available
    else:
        metrics = [m for m in metrics if m in available]

    if not metrics:
        print("No hay métricas para mostrar")
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        mask = df[metric].notna()
        ax.bar(df[mask]["config"], df[mask][metric])
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig