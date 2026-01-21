"""
Módulo para visualización UMAP
"""
import numpy as np
import pandas as pd
import visualization.my_umap as my_umap
import plotly.express as px
import plotly.graph_objects as go


def create_umap_projection(X, n_neighbors=30, min_dist=0.1, n_components=2, 
                          metric='euclidean', random_state=42, **kwargs):
    """
    Crea proyección UMAP de los datos
    
    Parameters
    ----------
    X : array-like
        Datos a proyectar
    n_neighbors : int, default=30
        Número de vecinos para UMAP
    min_dist : float, default=0.1
        Distancia mínima entre puntos
    n_components : int, default=2
        Número de dimensiones de salida
    metric : str, default='euclidean'
        Métrica de distancia
    random_state : int, default=42
        Semilla aleatoria
    **kwargs : dict
        Argumentos adicionales para UMAP
        
    Returns
    -------
    X_umap : ndarray
        Datos proyectados
    reducer : umap.UMAP
        Objeto UMAP ajustado
    """
    reducer = my_umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        **kwargs
    )
    
    X_umap = reducer.fit_transform(X)
    
    return X_umap, reducer


def plot_umap_labels(X, y, label_names=None, color_map=None,
                    n_neighbors=30, min_dist=0.1, metric='euclidean',
                    hover_data=None, title="UMAP Projection",
                    width=900, height=700, random_state=42,
                    save_path=None, **umap_kwargs):
    """
    Visualiza proyección UMAP coloreada por etiquetas verdaderas
    
    Parameters
    ----------
    X : array-like
        Datos a proyectar
    y : array-like
        Etiquetas verdaderas (0/1 o categorías)
    label_names : dict, optional
        Mapeo de valores de y a nombres. Ej: {0: 'Normal', 1: 'Attack'}
    color_map : dict, optional
        Mapeo de nombres a colores. Ej: {'Normal': '#1f77b4', 'Attack': '#ff7f0e'}
    n_neighbors : int, default=30
        Número de vecinos para UMAP
    min_dist : float, default=0.1
        Distancia mínima entre puntos
    metric : str, default='euclidean'
        Métrica de distancia
    hover_data : list of str or DataFrame columns, optional
        Columnas adicionales para mostrar en hover
    title : str, default="UMAP Projection"
        Título del gráfico
    width : int, default=900
        Ancho de la figura
    height : int, default=700
        Alto de la figura
    random_state : int, default=42
        Semilla aleatoria
    save_path : str, optional
        Ruta para guardar la imagen
    **umap_kwargs : dict
        Argumentos adicionales para UMAP
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figura de Plotly
    X_umap : ndarray
        Datos proyectados
    """
    # Crear proyección UMAP
    X_umap, reducer = create_umap_projection(
        X, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **umap_kwargs
    )
    
    # Crear DataFrame para visualización
    df_plot = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1],
        'label_value': y
    })
    
    # Mapear etiquetas a nombres
    if label_names is None:
        # Si y es binario, usar nombres por defecto
        unique_labels = np.unique(y)
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            label_names = {0: 'Class 0', 1: 'Class 1'}
        else:
            label_names = {val: f'Class {val}' for val in unique_labels}
    
    df_plot['Label'] = df_plot['label_value'].map(label_names)
    
    # Agregar datos adicionales para hover si se proporcionan
    hover_cols = ['label_value']
    if hover_data is not None:
        if isinstance(hover_data, pd.DataFrame):
            for col in hover_data.columns:
                df_plot[col] = hover_data[col].values
                hover_cols.append(col)
        elif isinstance(hover_data, list):
            hover_cols.extend(hover_data)
    
    # Crear gráfico
    fig = px.scatter(
        df_plot,
        x='UMAP1',
        y='UMAP2',
        color='Label',
        color_discrete_map=color_map,
        hover_data=hover_cols,
        title=title,
        width=width,
        height=height
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(legend_title_text='Class')
    
    # Guardar si se especifica ruta
    if save_path is not None:
        fig.write_image(save_path, width=width, height=height, scale=2)
    
    return fig, X_umap


def plot_umap_clusters(X, cluster_labels, hover_data=None,
                      n_neighbors=30, min_dist=0.1, metric='euclidean',
                      title="UMAP + Clustering",
                      width=900, height=700, random_state=42,
                      save_path=None, **umap_kwargs):
    """
    Visualiza proyección UMAP coloreada por clusters (soporta outliers como -1)
    
    Parameters
    ----------
    X : array-like
        Datos a proyectar
    cluster_labels : array-like
        Etiquetas de clusters (-1 para outliers)
    hover_data : list of str or DataFrame columns, optional
        Columnas adicionales para mostrar en hover
    n_neighbors : int, default=30
        Número de vecinos para UMAP
    min_dist : float, default=0.1
        Distancia mínima entre puntos
    metric : str, default='euclidean'
        Métrica de distancia
    title : str, default="UMAP + Clustering"
        Título del gráfico
    width : int, default=900
        Ancho de la figura
    height : int, default=700
        Alto de la figura
    random_state : int, default=42
        Semilla aleatoria
    save_path : str, optional
        Ruta para guardar la imagen
    **umap_kwargs : dict
        Argumentos adicionales para UMAP
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figura de Plotly
    X_umap : ndarray
        Datos proyectados
    """
    # Crear proyección UMAP
    X_umap, reducer = create_umap_projection(
        X,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **umap_kwargs
    )
    
    # Crear DataFrame para visualización
    df_plot = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1],
        'cluster_label': cluster_labels
    })
    
    # Mapear clusters a nombres ordenados
    df_plot['Cluster'] = df_plot['cluster_label'].map(
        lambda lbl: 'Outlier' if lbl == -1 else f'Cluster {lbl}'
    )
    
    # Ordenar categorías: Outlier primero, luego clusters
    cluster_order = ['Outlier'] + [
        f'Cluster {i}' 
        for i in sorted(df_plot[df_plot['cluster_label'] != -1]['cluster_label'].unique())
    ]
    df_plot['Cluster'] = pd.Categorical(
        df_plot['Cluster'], 
        categories=cluster_order, 
        ordered=True
    )
    
    # Agregar datos adicionales para hover
    hover_cols = ['cluster_label']
    if hover_data is not None:
        if isinstance(hover_data, pd.DataFrame):
            for col in hover_data.columns:
                df_plot[col] = hover_data[col].values
                hover_cols.append(col)
        elif isinstance(hover_data, list):
            hover_cols.extend(hover_data)
    
    # Crear gráfico
    fig = px.scatter(
        df_plot,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_data=hover_cols,
        title=title,
        width=width,
        height=height
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(legend_title_text='Clusters')
    
    # Guardar si se especifica ruta
    if save_path is not None:
        fig.write_image(save_path, width=width, height=height, scale=2)
    
    return fig, X_umap


def plot_umap_comparison(X, y, cluster_labels, label_names=None,
                        n_neighbors=30, min_dist=0.1, metric='euclidean',
                        width=1800, height=700, random_state=42,
                        save_path=None):
    """
    Visualiza comparación lado a lado: etiquetas verdaderas vs clusters
    
    Parameters
    ----------
    X : array-like
        Datos a proyectar
    y : array-like
        Etiquetas verdaderas
    cluster_labels : array-like
        Etiquetas de clusters
    label_names : dict, optional
        Mapeo de valores de y a nombres
    n_neighbors : int, default=30
        Número de vecinos para UMAP
    min_dist : float, default=0.1
        Distancia mínima entre puntos
    metric : str, default='euclidean'
        Métrica de distancia
    width : int, default=1800
        Ancho total de la figura
    height : int, default=700
        Alto de la figura
    random_state : int, default=42
        Semilla aleatoria
    save_path : str, optional
        Ruta para guardar la imagen
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figura con subplots
    X_umap : ndarray
        Datos proyectados
    """
    from plotly.subplots import make_subplots
    
    # Crear proyección UMAP (una sola vez)
    X_umap, reducer = create_umap_projection(
        X,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    # Preparar datos
    if label_names is None:
        unique_labels = np.unique(y)
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            label_names = {0: 'Normal', 1: 'Attack'}
        else:
            label_names = {val: f'Class {val}' for val in unique_labels}
    
    # Crear subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Ground Truth Labels", "Cluster Labels"),
        horizontal_spacing=0.1
    )
    
    # Plot 1: Etiquetas verdaderas
    for label_val in np.unique(y):
        mask = y == label_val
        fig.add_trace(
            go.Scatter(
                x=X_umap[mask, 0],
                y=X_umap[mask, 1],
                mode='markers',
                name=label_names[label_val],
                marker=dict(size=6, opacity=0.8),
                legendgroup='ground_truth',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot 2: Clusters
    for cluster_val in np.unique(cluster_labels):
        mask = cluster_labels == cluster_val
        cluster_name = 'Outlier' if cluster_val == -1 else f'Cluster {cluster_val}'
        fig.add_trace(
            go.Scatter(
                x=X_umap[mask, 0],
                y=X_umap[mask, 1],
                mode='markers',
                name=cluster_name,
                marker=dict(size=6, opacity=0.8),
                legendgroup='clusters',
                showlegend=True
            ),
            row=1, col=2
        )
    
    # Actualizar layout
    fig.update_xaxes(title_text="UMAP1", row=1, col=1)
    fig.update_xaxes(title_text="UMAP1", row=1, col=2)
    fig.update_yaxes(title_text="UMAP2", row=1, col=1)
    fig.update_yaxes(title_text="UMAP2", row=1, col=2)
    
    fig.update_layout(
        title_text="UMAP Projection Comparison",
        width=width,
        height=height
    )
    
    # Guardar si se especifica ruta
    if save_path is not None:
        fig.write_image(save_path, width=width, height=height, scale=2)
    
    return fig, X_umap