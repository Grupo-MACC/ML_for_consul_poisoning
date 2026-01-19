"""
Interactive Storytelling Application for Consul Poisoning Detection
=========================================================================
This Dash application allows interactive exploration of:
- Exploratory Data Analysis (EDA)
- Clustering model performance
- Prediction explanations
"""

import hdbscan
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    homogeneity_score, completeness_score, v_measure_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND DATA LOADING
# ============================================================================

# Data and model paths
DATA_PATH = 'data/processed/windowed_dataset_cleaned.csv'
MODELS_PATH = {
    'gmm': 'models/gmm_model.joblib',
    'hdbscan': 'models/hdbscan_outlier_model.joblib',
    'isolation_forest': 'models/isolation_forest_model.joblib'
}

# Corporate colors
COLORS = {
    'normal': '#2ecc71',
    'attack': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#9b59b6',
    'warning': '#f39c12',
    'neutral': '#95a5a6',
    'background': '#f8f9fa',
    'card': '#ffffff'
}

# Load data
print("📊 Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['is_attack'])
y = df['is_attack']
feature_names = X.columns.tolist()

# Load models
print("🤖 Loading models...")
models = {}
for name, path in MODELS_PATH.items():
    try:
        models[name] = joblib.load(path)
        print(f"  ✓ {name} loaded")
    except Exception as e:
        print(f"  ✗ Error loading {name}: {e}")

# Pre-calculate predictions
print("🔮 Calculating predictions...")
predictions = {}
for name, model in models.items():
    try:
        if name == 'isolation_forest':
            # Isolation Forest may have issues with feature names
            try:
                predictions[name] = model.predict(X)
            except ValueError:
                # If there's a problem with names, use numpy values
                predictions[name] = model.predict(X.values)
            # Convert -1 (anomaly) to 1, 1 (normal) to 0
            predictions[name] = np.where(predictions[name] == -1, 1, 0)
        elif name == 'hdbscan':
            predictions[name] = model.fit_predict(X)
        else:  # gmm
            predictions[name] = model.fit_predict(X)
        print(f"  ✓ {name} predictions calculated")
    except Exception as e:
        print(f"  ✗ Error predicting {name}: {e}")

# Pre-calculate PCA for visualizations
print("📉 Calculating dimensionality reduction...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df_viz = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'is_attack': y,
    'label': y.map({0: 'Normal', 1: 'Attack'})
})

# Add predictions to visualization dataframe
for name, preds in predictions.items():
    df_viz[f'{name}_pred'] = preds

print("✅ Data and models loaded successfully!")

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)
app.title = 'Consul Poisoning Detection - Storytelling'
server = app.server

# ============================================================================
# REUSABLE COMPONENTS
# ============================================================================

def create_card(title, children, color='primary'):
    """Create a styled card"""
    return dbc.Card([
        dbc.CardHeader(html.H5(title, className='mb-0 text-white'), 
                    style={'backgroundColor': COLORS[color]}),
        dbc.CardBody(children)
    ], className='mb-4 shadow-sm')

def create_metric_card(title, value, subtitle='', icon='📊'):
    """Create a metric card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '2rem'}),
                html.H3(value, className='mb-0 mt-2'),
                html.P(title, className='text-muted mb-0'),
                html.Small(subtitle, className='text-muted')
            ], className='text-center')
        ])
    ], className='mb-3 shadow-sm h-100')

# ============================================================================
# SECTION: INTRODUCTION
# ============================================================================

def create_intro_section():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H2('🎯 Consul Poisoning Attack Detection', className='mb-4'),
                html.P([
                    'This analysis focuses on detecting anomalous behavior in a ',
                    html.Strong('microservices'), ' environment that uses ', html.Strong('Consul'), 
                    ' as a service registry and service discovery mechanism.'
                ], className='lead'),
                html.Hr(),
                dbc.Alert([
                    html.H5('What is Consul Poisoning?', className='alert-heading'),
                    html.P([
                        'It is an attack where an attacker sends malicious registration requests to Consul, ',
                        'impersonating legitimate services. This can cause legitimate traffic ',
                        'to be redirected to the attacker.'
                    ])
                ], color='warning', className='mb-4'),
            ], md=8),
            dbc.Col([
                create_metric_card('Total Samples', f'{len(df):,}', 'Time windows', '📊'),
            ], md=1),
            dbc.Col([
                create_metric_card('Features', f'{len(feature_names)}', 'Input variables', '🔢'),
            ], md=1),
                dbc.Col([
                create_metric_card('Attacks', f'{y.sum():,}', f'{(y.sum()/len(y)*100):.1f}% of total', '⚠️'),
            ], md=1)
        ]),
        dbc.Row([
            dbc.Col([
                create_card('Dataset Description', [
                    html.P([
                        'The dataset contains service registration request records to Consul, ',
                        'aggregated in time windows (sliding windows). Each instance represents ',
                        'the behavior of an IP during a time interval.'
                    ]),
                    html.Ul([
                        html.Li([html.Strong('Normal Instances: '), f'{(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)']),
                        html.Li([html.Strong('Attack Instances: '), f'{(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)']),
                        html.Li([html.Strong('Imbalance Ratio: '), f'{(y==0).sum()/(y==1).sum():.1f}:1']),
                    ])
                ], color='primary')
            ], md=6),
            dbc.Col([
                create_card('Detection Approach', [
                    html.Ul([
                        html.Li([html.Strong('Sliding Windows: '), 'Temporal traffic aggregation by IP']),
                        html.Li([html.Strong('Derived Features: '), 'Frequency, diversity, patterns']),
                        html.Li([html.Strong('Unsupervised Clustering: '), 'GMM, HDBSCAN, Isolation Forest']),
                        html.Li([html.Strong('Anomaly Detection: '), 'Identify deviations from normal behavior']),
                    ]),
                    dbc.Badge('Unsupervised Learning', color='info', className='me-2'),
                    dbc.Badge('Anomaly Detection', color='warning'),
                ], color='secondary')
            ], md=6)
        ])
    ])

# ============================================================================
# SECTION: EDA
# ============================================================================

def create_eda_section():
    return html.Div([
        html.H2('📈 Exploratory Data Analysis', className='mb-4'),
        
        # Controls
        dbc.Row([
            dbc.Col([
                dbc.Label('Select Feature:'),
                dcc.Dropdown(
                    id='eda-feature-dropdown',
                    options=[{'label': f, 'value': f} for f in feature_names],
                    value=feature_names[0],
                    clearable=False
                )
            ], md=4),
            dbc.Col([
                dbc.Label('Visualization Type:'),
                dcc.RadioItems(
                    id='eda-viz-type',
                    options=[
                        {'label': ' Histogram', 'value': 'histogram'},
                        {'label': ' Box Plot', 'value': 'box'},
                        {'label': ' Violin', 'value': 'violin'}
                    ],
                    value='histogram',
                    inline=True
                )
            ], md=4),
            dbc.Col([
                dbc.Label('Split by class:'),
                dbc.Switch(id='eda-split-class', value=True, label='Enable')
            ], md=4)
        ], className='mb-4'),
        
        # EDA Charts
        dbc.Row([
            dbc.Col([
                create_card('Class Distribution', [
                    dcc.Graph(id='class-distribution-plot')
                ]) 
            ], md=6),
            dbc.Col([
                create_card('Feature Distribution', [
                    dcc.Graph(id='feature-distribution-plot')
                ])
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Correlation Matrix (Top Features)', [
                    dcc.Graph(id='correlation-matrix')
                ])
            ], md=6),
            dbc.Col([
                create_card('Descriptive Statistics', [
                    html.Div(id='stats-table')
                ])
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Feature Comparison by Class', [
                    dcc.Graph(id='feature-comparison-plot')
                ])
            ])
        ])
    ])

# ============================================================================
# SECTION: 2D VISUALIZATION
# ============================================================================

def create_visualization_section():
    return html.Div([
        html.H2('🗺️ Feature Space Visualization', className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Label('Color by:'),
                dcc.Dropdown(
                    id='viz-color-by',
                    options=[
                        {'label': 'Real Label (is_attack)', 'value': 'label'},
                        {'label': 'GMM Prediction', 'value': 'gmm_pred'},
                        {'label': 'HDBSCAN Prediction', 'value': 'hdbscan_pred'},
                        {'label': 'Isolation Forest Prediction', 'value': 'isolation_forest_pred'}
                    ],
                    value='label',
                    clearable=False
                )
            ], md=4),
            dbc.Col([
                dbc.Label('Sample size:'),
                dcc.Slider(
                    id='viz-sample-size',
                    min=1000,
                    max=len(df),
                    step=1000,
                    value=min(5000, len(df)),
                    marks={1000: '1K', 5000: '5K', 10000: '10K', len(df): 'All'}
                )
            ], md=8)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                create_card('PCA 2D Projection', [
                    dcc.Graph(id='pca-scatter-plot', style={'height': '500px'})
                ])
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Explained Variance by PCA Components', [
                    dcc.Graph(id='pca-variance-plot')
                ])
            ], md=6),
            dbc.Col([
                create_card('Principal Components Distribution', [
                    dcc.Graph(id='pca-distribution-plot')
                ])
            ], md=6)
        ])
    ])

# ============================================================================
# SECTION: MODEL PERFORMANCE
# ============================================================================

def create_model_section():
    return html.Div([
        html.H2('🤖 Model Performance', className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Label('Select Model:'),
                dcc.Dropdown(
                    id='model-selector',
                    options=[
                        {'label': 'GMM (Gaussian Mixture Model)', 'value': 'gmm'},
                        {'label': 'HDBSCAN (Density-based)', 'value': 'hdbscan'},
                    ],
                    value='gmm',
                    clearable=False
                )
            ], md=6)
        ], className='mb-4'),
        
        # Model metrics
        dbc.Row([
            dbc.Col([html.Div(id='model-metrics-cards')], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Confusion Matrix (Semi-supervised)', [
                    dcc.Graph(id='confusion-matrix-plot')
                ])
            ], md=6),
            dbc.Col([
                create_card('Clustering Metrics', [
                    dcc.Graph(id='metrics-bar-plot')
                ])
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Model Comparison', [
                    dcc.Graph(id='model-comparison-plot')
                ])
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Cluster Distribution by Model', [
                    dcc.Graph(id='cluster-distribution-plot')
                ])
            ])
        ])
    ])

# ============================================================================
# SECTION: EXPLANATIONS
# ============================================================================

def create_explanation_section():
    return html.Div([
        html.H2('🔍 Prediction Explanations', className='mb-4'),
        
        dbc.Alert([
            html.H5('Model Interpretability', className='alert-heading'),
            html.P([
                'This section analyzes which features are most important for distinguishing ',
                'between normal and potentially malicious behavior.'
            ])
        ], color='info', className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                create_card('Feature Importance (Mean Difference)', [
                    dcc.Graph(id='feature-importance-plot')
                ])
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Top Discriminative Features', [
                    dcc.Graph(id='top-features-plot')
                ])
            ], md=6),
            dbc.Col([
                create_card('Individual Instance Analysis', [
                    dbc.Label('Select sample index:'),
                    dcc.Input(
                        id='sample-index-input',
                        type='number',
                        min=0,
                        max=len(df)-1,
                        value=0,
                        className='mb-3'
                    ),
                    html.Div(id='sample-analysis')
                ])
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card('Attack vs Normal Patterns', [
                    dcc.Graph(id='attack-patterns-plot')
                ])
            ])
        ])
    ])

# ============================================================================
# MAIN LAYOUT
# ============================================================================

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("🏠 Introduction", id="nav-intro", href="#", n_clicks=0)),
        dbc.NavItem(dbc.NavLink("📊 EDA", id="nav-eda", href="#", n_clicks=0)),
        dbc.NavItem(dbc.NavLink("🗺️ Visualization", id="nav-viz", href="#", n_clicks=0)),
        dbc.NavItem(dbc.NavLink("🤖 Models", id="nav-models", href="#", n_clicks=0)),
        dbc.NavItem(dbc.NavLink("🔍 Explanations", id="nav-explain", href="#", n_clicks=0)),
    ],
    brand="🛡️ Consul Poisoning Detection",
    brand_href="#",
    color="primary",
    dark=True,
    sticky="top"
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        # Navigation Tabs
        dcc.Tabs(id='main-tabs', value='tab-intro', children=[
            dcc.Tab(label='🏠 Introduction', value='tab-intro', children=[
                html.Div(create_intro_section(), className='p-4')
            ]),
            dcc.Tab(label='📊 EDA', value='tab-eda', children=[
                html.Div(create_eda_section(), className='p-4')
            ]),
            dcc.Tab(label='🗺️ Visualization', value='tab-viz', children=[
                html.Div(create_visualization_section(), className='p-4')
            ]),
            dcc.Tab(label='🤖 Models', value='tab-models', children=[
                html.Div(create_model_section(), className='p-4')
            ]),
            dcc.Tab(label='🔍 Explanations', value='tab-explain', children=[
                html.Div(create_explanation_section(), className='p-4')
            ]),
        ], className='mt-4'),
        
        # Footer
        html.Hr(),
        html.Footer([
            html.P('Consul Poisoning Detection - ML Analysis for Security', 
                className='text-center text-muted')
        ], className='mb-4')
    ], fluid=True)
], style={'backgroundColor': COLORS['background']})

# ============================================================================
# CALLBACKS
# ============================================================================

# Callback: Navbar -> Tabs
@app.callback(
    Output('main-tabs', 'value'),
    [Input('nav-intro', 'n_clicks'),
     Input('nav-eda', 'n_clicks'),
     Input('nav-viz', 'n_clicks'),
     Input('nav-models', 'n_clicks'),
     Input('nav-explain', 'n_clicks')],
    prevent_initial_call=True
)
def navigate_tabs(n_intro, n_eda, n_viz, n_models, n_explain):
    from dash import ctx
    triggered_id = ctx.triggered_id
    
    tab_map = {
        'nav-intro': 'tab-intro',
        'nav-eda': 'tab-eda',
        'nav-viz': 'tab-viz',
        'nav-models': 'tab-models',
        'nav-explain': 'tab-explain'
    }
    
    return tab_map.get(triggered_id, 'tab-intro')

# Callback: Class distribution
@app.callback(
    Output('class-distribution-plot', 'figure'),
    Input('eda-feature-dropdown', 'value')  # Trigger on load
)
def update_class_distribution(_):
    counts = y.value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=['Normal', 'Attack'],
            y=counts.values,
            marker_color=[COLORS['normal'], COLORS['attack']],
            text=[f'{v:,} ({v/len(y)*100:.1f}%)' for v in counts.values],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Class',
        yaxis_title='Number of Samples',
        showlegend=False,
        template='plotly_white'
    )
    return fig

# Callback: Feature distribution
@app.callback(
    Output('feature-distribution-plot', 'figure'),
    [Input('eda-feature-dropdown', 'value'),
     Input('eda-viz-type', 'value'),
     Input('eda-split-class', 'value')]
)
def update_feature_distribution(feature, viz_type, split_class):
    if split_class:
        color_col = 'label'
        colors = {'Normal': COLORS['normal'], 'Attack': COLORS['attack']}
    else:
        color_col = None
        colors = None
    
    df_plot = df.copy()
    df_plot['label'] = df_plot['is_attack'].map({0: 'Normal', 1: 'Attack'})
    
    if viz_type == 'histogram':
        fig = px.histogram(df_plot, x=feature, color=color_col if split_class else None,
                          color_discrete_map=colors, barmode='overlay', opacity=0.7)
    elif viz_type == 'box':
        fig = px.box(df_plot, y=feature, x='label' if split_class else None,
                    color='label' if split_class else None, color_discrete_map=colors)
    else:  # violin
        fig = px.violin(df_plot, y=feature, x='label' if split_class else None,
                       color='label' if split_class else None, color_discrete_map=colors, box=True)
    
    fig.update_layout(
        title=f'Distribution of {feature}',
        template='plotly_white'
    )
    return fig

# Callback: Correlation matrix
@app.callback(
    Output('correlation-matrix', 'figure'),
    Input('eda-feature-dropdown', 'value')
)
def update_correlation_matrix(_):
    # Select top 15 features by variance
    top_features = X.var().nlargest(15).index.tolist()
    corr_matrix = X[top_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=top_features,
        y=top_features,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={'size': 8}
    ))
    fig.update_layout(
        title='Correlation Matrix (Top 15 Features)',
        template='plotly_white',
        height=500
    )
    return fig

# Callback: Descriptive statistics
@app.callback(
    Output('stats-table', 'children'),
    Input('eda-feature-dropdown', 'value')
)
def update_stats_table(feature):
    stats_normal = X[y == 0][feature].describe()
    stats_attack = X[y == 1][feature].describe()
    
    stats_df = pd.DataFrame({
        'Statistic': stats_normal.index,
        'Normal': stats_normal.values.round(4),
        'Attack': stats_attack.values.round(4)
    })
    
    return dbc.Table.from_dataframe(
        stats_df, striped=True, bordered=True, hover=True, size='sm'
    )

# Callback: Feature comparison
@app.callback(
    Output('feature-comparison-plot', 'figure'),
    Input('eda-feature-dropdown', 'value')
)
def update_feature_comparison(_):
    # Calculate normalized mean differences
    means_normal = X[y == 0].mean()
    means_attack = X[y == 1].mean()
    stds = X.std()
    
    diff = ((means_attack - means_normal) / stds).sort_values()
    top_diff = pd.concat([diff.head(10), diff.tail(10)])
    
    colors = [COLORS['normal'] if v < 0 else COLORS['attack'] for v in top_diff.values]
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_diff.index,
            x=top_diff.values,
            orientation='h',
            marker_color=colors
        )
    ])
    fig.update_layout(
        title='Normalized Mean Difference (Attack vs Normal)',
        xaxis_title='Standardized Difference',
        yaxis_title='Feature',
        template='plotly_white',
        height=500
    )
    return fig

# Callback: PCA Scatter
@app.callback(
    Output('pca-scatter-plot', 'figure'),
    [Input('viz-color-by', 'value'),
     Input('viz-sample-size', 'value')]
)
def update_pca_scatter(color_by, sample_size):
    # Subsample if needed
    if sample_size < len(df_viz):
        df_sample = df_viz.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_viz
    
    if color_by == 'label':
        fig = px.scatter(
            df_sample, x='PC1', y='PC2', color='label',
            color_discrete_map={'Normal': COLORS['normal'], 'Attack': COLORS['attack']},
            opacity=0.6,
            title='PCA Projection - Real Labels'
        )
    else:
        fig = px.scatter(
            df_sample, x='PC1', y='PC2', color=color_by,
            opacity=0.6,
            title=f'PCA Projection - {color_by.replace("_pred", "").upper()} Prediction'
        )
    
    fig.update_layout(template='plotly_white')
    fig.update_traces(marker={'size': 5})
    return fig

# Callback: PCA Variance
@app.callback(
    Output('pca-variance-plot', 'figure'),
    Input('viz-color-by', 'value')
)
def update_pca_variance(_):
    pca_full = PCA(n_components=10, random_state=42)
    pca_full.fit(X)
    
    var_explained = pca_full.explained_variance_ratio_ * 100
    cumsum = np.cumsum(var_explained)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=[f'PC{i+1}' for i in range(10)], y=var_explained, name='Individual'),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=[f'PC{i+1}' for i in range(10)], y=cumsum, 
                name='Cumulative', mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Explained Variance by PCA Components',
        template='plotly_white'
    )
    fig.update_yaxes(title_text='Individual Variance (%)', secondary_y=False)
    fig.update_yaxes(title_text='Cumulative Variance (%)', secondary_y=True)
    
    return fig

# Callback: PCA Distribution
@app.callback(
    Output('pca-distribution-plot', 'figure'),
    Input('viz-color-by', 'value')
)
def update_pca_distribution(_):
    df_plot = df_viz.copy()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['PC1', 'PC2'])
    
    for i, (label, color) in enumerate([('Normal', COLORS['normal']), ('Attack', COLORS['attack'])]):
        mask = df_plot['label'] == label
        fig.add_trace(
            go.Histogram(x=df_plot[mask]['PC1'], name=label, 
                        marker_color=color, opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=df_plot[mask]['PC2'], name=label, 
                        marker_color=color, opacity=0.7, showlegend=False),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Principal Components Distribution',
        template='plotly_white',
        barmode='overlay'
    )
    return fig

# Callback: Model metrics
@app.callback(
    Output('model-metrics-cards', 'children'),
    Input('model-selector', 'value')
)
def update_model_metrics(model_name):
    if model_name not in predictions:
        return html.P('Model not available')
    
    preds = predictions[model_name]
    
    # Calculate metrics
    try:
        if model_name == 'hdbscan':
            mask = preds != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], preds[mask], random_state=42)
                homogeneity = homogeneity_score(y[mask], preds[mask])
                completeness = completeness_score(y[mask], preds[mask])
                v_measure = v_measure_score(y[mask], preds[mask])
            else:
                silhouette = homogeneity = completeness = v_measure = 0
            n_clusters = len(np.unique(preds[preds != -1]))
            n_noise = (preds == -1).sum()
        else:
            silhouette = silhouette_score(X, preds, random_state=42) if len(np.unique(preds)) > 1 else 0
            homogeneity = homogeneity_score(y, preds)
            completeness = completeness_score(y, preds)
            v_measure = v_measure_score(y, preds)
            n_clusters = len(np.unique(preds))
            n_noise = 0
    except Exception as e:
        silhouette = homogeneity = completeness = v_measure = 0
        n_clusters = n_noise = 0
    
    return dbc.Row([
        dbc.Col(create_metric_card('Silhouette', f'{silhouette:.3f}', 'Internal cohesion', '📏')),
        dbc.Col(create_metric_card('Homogeneity', f'{homogeneity:.3f}', 'Cluster purity', '🎯')),
        dbc.Col(create_metric_card('Completeness', f'{completeness:.3f}', 'Class coverage', '✅')),
        dbc.Col(create_metric_card('V-Measure', f'{v_measure:.3f}', 'H/C Balance', '⚖️')),
        dbc.Col(create_metric_card('Clusters', f'{n_clusters}', f'{n_noise} noise points', '🔢')),
    ])

# Callback: Confusion matrix
@app.callback(
    Output('confusion-matrix-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_confusion_matrix(model_name):
    if model_name not in predictions:
        return go.Figure()
    
    preds = predictions[model_name]
    cm_df = pd.crosstab(y, preds, rownames=['Real'], colnames=['Prediction'])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=[f'Cluster {c}' for c in cm_df.columns],
        y=['Normal (0)', 'Attack (1)'],
        colorscale='Blues',
        text=cm_df.values,
        texttemplate='%{text}',
        textfont={'size': 14}
    ))
    fig.update_layout(
        title=f'Confusion Matrix - {model_name.upper()}',
        template='plotly_white'
    )
    return fig

# Callback: Bar metrics
@app.callback(
    Output('metrics-bar-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_metrics_bar(model_name):
    if model_name not in predictions:
        return go.Figure()
    
    preds = predictions[model_name]
    
    try:
        if model_name == 'hdbscan':
            mask = preds != -1
            metrics = {
                'Homogeneity': homogeneity_score(y[mask], preds[mask]),
                'Completeness': completeness_score(y[mask], preds[mask]),
                'V-Measure': v_measure_score(y[mask], preds[mask]),
                'Silhouette': silhouette_score(X[mask], preds[mask], random_state=42) if len(np.unique(preds[mask])) > 1 else 0
            }
        else:
            metrics = {
                'Homogeneity': homogeneity_score(y, preds),
                'Completeness': completeness_score(y, preds),
                'V-Measure': v_measure_score(y, preds),
                'Silhouette': silhouette_score(X, preds, random_state=42) if len(np.unique(preds)) > 1 else 0
            }
    except:
        metrics = {'Homogeneity': 0, 'Completeness': 0, 'V-Measure': 0, 'Silhouette': 0}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=[COLORS['primary'], COLORS['secondary'], 
                         COLORS['warning'], COLORS['normal']],
            text=[f'{v:.3f}' for v in metrics.values()],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title=f'Clustering Metrics - {model_name.upper()}',
        yaxis_range=[0, 1],
        template='plotly_white'
    )
    return fig

# Callback: Model comparison
@app.callback(
    Output('model-comparison-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_model_comparison(_):
    comparison_data = []
    
    for model_name, preds in predictions.items():
        try:
            if model_name == "isolation_forest":
                break
            if model_name == 'hdbscan':
                mask = preds != -1
                if mask.sum() > 0:
                    h = homogeneity_score(y[mask], preds[mask])
                    c = completeness_score(y[mask], preds[mask])
                    v = v_measure_score(y[mask], preds[mask])
                    s = silhouette_score(X[mask], preds[mask], random_state=42) if len(np.unique(preds[mask])) > 1 else 0
                else:
                    h = c = v = s = 0
            else:
                h = homogeneity_score(y, preds)
                c = completeness_score(y, preds)
                v = v_measure_score(y, preds)
                s = silhouette_score(X, preds, random_state=42) if len(np.unique(preds)) > 1 else 0
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Homogeneity': h, 'Completeness': c, 
                'V-Measure': v, 'Silhouette': s
            })
        except:
            pass
    
    df_comp = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    for metric in ['Homogeneity', 'Completeness', 'V-Measure', 'Silhouette']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_comp['Model'],
            y=df_comp[metric],
            text=[f'{v:.3f}' for v in df_comp[metric]],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Model Comparison',
        barmode='group',
        yaxis_range=[0, 1],
        template='plotly_white'
    )
    return fig

# Callback: Cluster distribution
@app.callback(
    Output('cluster-distribution-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_cluster_distribution(model_name):
    fig = make_subplots(rows=1, cols=len(predictions), 
                    subplot_titles=[m.upper() for m in predictions.keys()])
    
    for i, (name, preds) in enumerate(predictions.items(), 1):
        if name == "isolation_forest":
            break
        cluster_counts = pd.Series(preds).value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=[f'C{c}' for c in cluster_counts.index], 
                y=cluster_counts.values, name=name),
            row=1, col=i
        )
    
    fig.update_layout(
        title='Sample Distribution by Cluster',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    return fig

# Callback: Feature importance
@app.callback(
    Output('feature-importance-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_feature_importance(_):
    # Calculate importance based on normalized mean difference
    means_normal = X[y == 0].mean()
    means_attack = X[y == 1].mean()
    stds = X.std().replace(0, 1)
    
    importance = np.abs((means_attack - means_normal) / stds).sort_values(ascending=False)
    top_20 = importance.head(20)
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_20.index,
            x=top_20.values,
            orientation='h',
            marker_color=COLORS['primary']
        )
    ])
    fig.update_layout(
        title='Top 20 Most Discriminative Features',
        xaxis_title='|Standardized Difference|',
        yaxis_title='Feature',
        template='plotly_white',
        height=600
    )
    return fig

# Callback: Top features
@app.callback(
    Output('top-features-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_top_features(_):
    # Top 5 features by mean difference
    means_normal = X[y == 0].mean()
    means_attack = X[y == 1].mean()
    stds = X.std().replace(0, 1)
    
    importance = np.abs((means_attack - means_normal) / stds).sort_values(ascending=False)
    top_5 = importance.head(5).index.tolist()
    
    df_plot = df[top_5 + ['is_attack']].copy()
    df_plot['Class'] = df_plot['is_attack'].map({0: 'Normal', 1: 'Attack'})
    df_melt = df_plot.melt(id_vars=['Class'], value_vars=top_5, 
                        var_name='Feature', value_name='Value')
    
    fig = px.box(df_melt, x='Feature', y='Value', color='Class',
                color_discrete_map={'Normal': COLORS['normal'], 'Attack': COLORS['attack']})
    fig.update_layout(
        title='Top 5 Discriminative Features',
        template='plotly_white'
    )
    return fig

# Callback: Individual sample analysis
@app.callback(
    Output('sample-analysis', 'children'),
    Input('sample-index-input', 'value')
)
def update_sample_analysis(idx):
    if idx is None or idx < 0 or idx >= len(df):
        return html.P('Invalid index')
    
    sample = X.iloc[idx]
    true_label = 'Attack' if y.iloc[idx] == 1 else 'Normal'
    
    # Top 5 most relevant features for this sample
    means_normal = X[y == 0].mean()
    stds = X.std().replace(0, 1)
    
    deviations = ((sample - means_normal) / stds).abs().sort_values(ascending=False)
    top_5 = deviations.head(5)
    
    return html.Div([
        dbc.Alert(f'Real label: {true_label}', 
                 color='danger' if true_label == 'Attack' else 'success'),
        html.H6('Top 5 Features with highest deviation from normal:'),
        dbc.Table([
            html.Thead(html.Tr([html.Th('Feature'), html.Th('Value'), html.Th('Deviation')])),
            html.Tbody([
                html.Tr([
                    html.Td(feat),
                    html.Td(f'{sample[feat]:.4f}'),
                    html.Td(f'{dev:.2f}σ')
                ]) for feat, dev in top_5.items()
            ])
        ], striped=True, bordered=True, size='sm')
    ])

# Callback: Attack patterns
@app.callback(
    Output('attack-patterns-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_attack_patterns(_):
    # Radar chart comparing top feature means
    means_normal = X[y == 0].mean()
    means_attack = X[y == 1].mean()
    stds = X.std().replace(0, 1)
    
    importance = np.abs((means_attack - means_normal) / stds).sort_values(ascending=False)
    top_10 = importance.head(10).index.tolist()
    
    # Normalize for radar
    normal_vals = ((means_normal[top_10] - X[top_10].min()) / 
                  (X[top_10].max() - X[top_10].min())).values
    attack_vals = ((means_attack[top_10] - X[top_10].min()) / 
                  (X[top_10].max() - X[top_10].min())).values
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(normal_vals) + [normal_vals[0]],
        theta=top_10 + [top_10[0]],
        fill='toself',
        name='Normal',
        line_color=COLORS['normal']
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(attack_vals) + [attack_vals[0]],
        theta=top_10 + [top_10[0]],
        fill='toself',
        name='Attack',
        line_color=COLORS['attack']
    ))
    
    fig.update_layout(
        title='Feature Profile: Normal vs Attack',
        polar={'radialaxis': {'visible': True, 'range': [0, 1]}},
        template='plotly_white'
    )
    return fig

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Dash application...")
    print("   Access at: http://127.0.0.]]1:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)
