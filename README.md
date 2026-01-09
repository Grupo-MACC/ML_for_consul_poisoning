# ML for Consul Poisoning Detection

A machine learning project to detect consul poisoning attacks using network data analysis and supervised/semi-supervised learning techniques.

## Quick Start

### Prerequisites
- Python 3.11
- Conda

### Installation

```bash
conda create --name consul-ml python=3.11
conda activate consul-ml
pip install -r requirements.txt
pip install -e .
```

If you encounter any issues, try reloading the environment:
```bash
conda deactivate
conda activate consul-ml
```

## Project Structure

- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for analysis and modeling
- `src/` - Source code for sliding window analysis and utilities
- `models/` - Trained model artifacts
- `reports/` - Analysis and results

## Dataset

The project uses network traffic data to identify consul poisoning attack patterns. Refer to [dataset-columns.md](dataset-columns.md) for detailed feature descriptions.

## Notebooks

1. `00_windowed_dataset_creation.ipynb` - Create sliding window dataset
2. `01_eda.ipynb` - Exploratory data analysis
3. `02_preprocessing.ipynb` - Data preprocessing and validation
4. `03_modeling.ipynb` - Model training and evaluation