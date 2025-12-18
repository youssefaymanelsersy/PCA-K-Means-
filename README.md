# PCA and K-Means Clustering

Implementations of PCA (Principal Component Analysis) and K-Means Clustering from scratch using NumPy, applied to the Breast Cancer Wisconsin dataset.

## Features

- **PCA**: Dimensionality reduction via eigenvalue decomposition
- **K-Means**: Clustering with K-Means++ initialization
- **Metrics**: Custom evaluation metrics for clustering
- **Analysis**: Elbow curves, variance analysis, visualizations

## Setup

```bash
pip install numpy matplotlib pandas scikit-learn
python main.py
```

## Output

Generates visualizations in `visualizations/`:

- PCA variance analysis
- K-Means elbow curves
- 2D cluster projections
- Clustering comparisons

## Files

- `pca.py` - PCA implementation
- `kmeans.py` - K-Means implementation
- `metrics.py` - Evaluation metrics
- `main.py` - Main analysis script
- `visualizations/` - Generated plots
- `final_versions/` - Final code versions
