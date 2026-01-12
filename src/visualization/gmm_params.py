"""
Módulo para visualización de Gaussian Mixture Models
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_samples, 
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)


def plot_gmm_silhouette(ax, X, labels, title, max_height=800):
    """
    Dibuja silhouette para clusters de GMM
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde dibujar
    X : array-like
        Datos originales
    labels : array-like
        Etiquetas de clustering
    title : str
        Título del gráfico
    max_height : int, default=800
        Altura máxima para submuestreo visual
        
    Returns
    -------
    float or None
        Silhouette score promedio, o None si hay menos de 2 clusters
    """
    # GMM no tiene outliers, pero verificamos que haya al menos 2 clusters
    if len(np.unique(labels)) < 2:
        ax.set_title(f"{title}\n<2 clusters")
        ax.axis("off")
        return None

    sil_vals = silhouette_samples(X, labels)
    sil_avg = silhouette_score(X, labels)

    y_lower = 10

    for cluster in np.unique(labels):
        vals = sil_vals[labels == cluster]
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


def compare_gmm_params(X, param_grid, y_true=None, covariance_type='full', 
                      reg_covar=1e-6, random_state=42, n_cols=2, figsize_per_row=4):
    """
    Compara diferentes configuraciones de GMM usando silhouette plots
    
    Parameters
    ----------
    X : array-like
        Datos a clusterizar
    param_grid : list of int
        Lista de n_components a probar
        Ejemplo: [2, 3, 4, 5, 6]
    y_true : array-like, optional
        Etiquetas verdaderas para calcular métricas supervisadas
    covariance_type : str, default='full'
        Tipo de covarianza: 'full', 'tied', 'diag', 'spherical'
    reg_covar : float, default=1e-6
        Regularización de covarianza
    random_state : int, default=42
        Semilla aleatoria
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
    n_rows = int(np.ceil(len(param_grid) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, figsize_per_row * n_rows),
        sharex=True
    )

    axes = axes.flatten()

    results = []

    for ax, n_comp in zip(axes, param_grid):
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state
        )

        labels = gmm.fit_predict(X)

        sil = plot_gmm_silhouette(
            ax,
            X,
            labels,
            title=f"n_components={n_comp}"
        )

        result = {
            "n_components": n_comp,
            "covariance_type": covariance_type,
            "silhouette": sil,
            "n_clusters": len(set(labels)),
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "log_likelihood": gmm.score(X) * len(X)  # score devuelve promedio
        }
        
        # Si tenemos etiquetas verdaderas, calcular métricas supervisadas
        if y_true is not None:
            result["homogeneity"] = homogeneity_score(y_true, labels)
            result["completeness"] = completeness_score(y_true, labels)
            result["v_measure"] = v_measure_score(y_true, labels)
        
        results.append(result)

    # Apagar ejes sobrantes
    for ax in axes[len(param_grid):]:
        ax.axis("off")

    plt.suptitle(f"Comparación de Silhouette – GMM (covariance_type='{covariance_type}')", 
                fontsize=14)
    plt.tight_layout()

    return results, fig


def plot_gmm_metrics_comparison(results, metrics=None, figsize=(14, 5)):
    """
    Visualiza comparación de métricas para diferentes configuraciones de GMM
    
    Parameters
    ----------
    results : list of dict
        Resultados de compare_gmm_params
    metrics : list of str, optional
        Métricas a visualizar. Por defecto usa todas las disponibles.
        Opciones: 'silhouette', 'homogeneity', 'completeness', 'v_measure', 
                 'bic', 'aic', 'log_likelihood'
    figsize : tuple, default=(14, 5)
        Tamaño de la figura
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Crear etiquetas limpias para el eje x
    df['config'] = df['n_components'].astype(int).astype(str)
    
    # Determinar qué métricas visualizar
    available_metrics = []
    if 'silhouette' in df.columns and df['silhouette'].notna().any():
        available_metrics.append('silhouette')
    if 'homogeneity' in df.columns and df['homogeneity'].notna().any():
        available_metrics.extend(['homogeneity', 'completeness', 'v_measure'])
    if 'bic' in df.columns:
        available_metrics.extend(['bic', 'aic'])
    
    if metrics is None:
        # Por defecto, mostrar métricas de clustering (no BIC/AIC)
        metrics = [m for m in available_metrics if m not in ['bic', 'aic', 'log_likelihood']]
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
        'homogeneity': '#2980b9',  # azul oscuro
        'completeness': '#e74c3c', # rojo vivo
        'v_measure': '#8e44ad',    # morado
        'bic': '#f39c12',          # naranja
        'aic': '#e67e22'           # naranja oscuro
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filtrar valores no nulos
        mask = df[metric].notna()
        x_vals = df[mask]['config']
        y_vals = df[mask][metric]
        
        bars = ax.bar(x_vals, y_vals, color=colors.get(metric, '#95a5a6'), 
                     alpha=0.85, edgecolor='black', linewidth=1)
        ax.set_xlabel('n_components', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.upper() if metric in ['bic', 'aic'] else metric.capitalize(), 
                     fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper() if metric in ["bic", "aic"] else metric.capitalize()} Score', 
                    fontsize=12, fontweight='bold')
        
        # Para BIC/AIC, menor es mejor (invertir escala visual)
        if metric in ['bic', 'aic']:
            # No establecer ylim fijo, dejarlo automático
            # Marcar el mínimo
            min_idx = y_vals.idxmin()
            bars[min_idx].set_color('#27ae60')
            bars[min_idx].set_alpha(0.9)
        else:
            ax.set_ylim(0, 1)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotar etiquetas si hay muchas
        if len(x_vals) > 4:
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        else:
            ax.tick_params(axis='x', labelsize=10)
        
        # Añadir valores sobre las barras
        for j, (x, y) in enumerate(zip(x_vals, y_vals)):
            if metric in ['bic', 'aic']:
                ax.text(j, y, f'{y:.1f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold', color='black', alpha=0.8)
            else:
                ax.text(j, y + 0.01, f'{y:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold', color='black', alpha=0.8)
    
    # Ocultar ejes sobrantes
    for ax in axes[len(metrics):]:
        ax.axis('off')
    
    plt.suptitle('Comparación de Métricas GMM por Configuración', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_gmm_information_criteria(results, figsize=(12, 5)):
    """
    Visualiza BIC y AIC para selección de modelo GMM
    Criterios de información: menor es mejor
    
    Parameters
    ----------
    results : list of dict
        Resultados de compare_gmm_params
    figsize : tuple, default=(12, 5)
        Tamaño de la figura
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    if 'bic' not in df.columns or 'aic' not in df.columns:
        print("No hay información de BIC/AIC disponible")
        return None
    
    # Crear etiquetas
    df['config'] = df['n_components'].astype(int).astype(str)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(df))
    width = 0.35
    
    # BIC
    bars1 = ax1.bar(x, df['bic'], width, label='BIC', 
                   color='#f39c12', alpha=0.85, edgecolor='black')
    ax1.set_xlabel('Configuración', fontsize=11, fontweight='bold')
    ax1.set_ylabel('BIC (menor es mejor)', fontsize=11, fontweight='bold')
    ax1.set_title('Bayesian Information Criterion', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Marcar el mejor (mínimo)
    min_bic_idx = df['bic'].idxmin()
    bars1[min_bic_idx].set_color('#27ae60')
    
    # AIC
    bars2 = ax2.bar(x, df['aic'], width, label='AIC', 
                   color='#e67e22', alpha=0.85, edgecolor='black')
    ax2.set_xlabel('Configuración', fontsize=11, fontweight='bold')
    ax2.set_ylabel('AIC (menor es mejor)', fontsize=11, fontweight='bold')
    ax2.set_title('Akaike Information Criterion', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Marcar el mejor (mínimo)
    min_aic_idx = df['aic'].idxmin()
    bars2[min_aic_idx].set_color('#27ae60')
    
    # Añadir valores
    for bars, data in [(bars1, df['bic']), (bars2, df['aic'])]:
        for bar, val in zip(bars, data):
            height = bar.get_height()
            ax = bar.axes
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    plt.suptitle('Criterios de Información para Selección de Modelo GMM', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def find_best_gmm(results, criterion='bic'):
    """
    Encuentra la mejor configuración GMM según un criterio
    
    Parameters
    ----------
    results : list of dict
        Resultados de compare_gmm_params
    criterion : str, default='bic'
        Criterio de selección: 'bic', 'aic', 'silhouette', 'v_measure'
        
    Returns
    -------
    dict
        Mejor configuración y sus métricas
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    if criterion not in df.columns:
        raise ValueError(f"Criterio '{criterion}' no disponible. Opciones: {df.columns.tolist()}")
    
    # Para BIC/AIC, menor es mejor; para el resto, mayor es mejor
    if criterion in ['bic', 'aic']:
        best_idx = df[criterion].idxmin()
    else:
        best_idx = df[criterion].idxmax()
    
    best_config = df.iloc[best_idx].to_dict()
    
    print(f"\n{'='*60}")
    print(f"Mejor configuración según {criterion.upper()}:")
    print(f"{'='*60}")
    print(f"n_components: {int(best_config['n_components'])}")
    print(f"covariance_type: {best_config['covariance_type']}")
    print(f"\nMétricas:")
    for key, val in best_config.items():
        if key not in ['n_components', 'covariance_type', 'config']:
            if val is not None:
                if key in ['bic', 'aic', 'log_likelihood']:
                    print(f"  {key}: {val:.2f}")
                else:
                    print(f"  {key}: {val:.4f}")
    print(f"{'='*60}\n")
    
    return best_config