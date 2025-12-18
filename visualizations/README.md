# Visualizations Folder Structure

This folder contains all generated visualizations organized by category.

## Folder Organization

### 📁 experiment_1/

Contains visualizations for **Experiment 1: K-Means on Original Data**

- `exp1_optimal_k_analysis.png` - 4-panel analysis showing:
  - Elbow curve
  - Silhouette scores
  - Gap statistic
  - K-Means++ vs Random initialization comparison

### 📁 experiment_3/

Contains visualizations for **Experiment 3: K-Means after PCA**

- `exp3_pca_analysis.png` - 6-panel analysis showing:
  - Variance explained by components
  - Reconstruction error
  - Silhouette scores vs components
  - Purity vs components
  - Davies-Bouldin Index
  - Execution time

### 📁 comparisons/

Contains comparison visualizations across experiments

- `confusion_matrices.png` - Confusion matrices for both experiments
- `clusters_2d_projections.png` - 3-panel cluster visualization (True labels, Exp 1, Exp 3)
- `heatmap_comparison.png` - Performance heatmap across all methods and metrics
- `clustering_comparison.png` - Comprehensive 4-panel comparison (from main.py)

### 📁 original_analysis/

Contains visualizations from the original main.py implementation

- `pca_variance_analysis.png` - PCA variance and reconstruction error
- `kmeans_elbow_curves.png` - Elbow curves for original and PCA data
- `kmeans_original_2d.png` - 2D clusters on original data
- `kmeans_pca_2d.png` - 2D clusters on PCA-reduced data

## How Visualizations are Generated

### From main_updated.py (Experiments 1 & 3)

```bash
python3 main_updated.py
```

Generates:

- `visualizations/experiment_1/exp1_optimal_k_analysis.png`
- `visualizations/experiment_3/exp3_pca_analysis.png`
- `visualizations/comparisons/confusion_matrices.png`
- `visualizations/comparisons/clusters_2d_projections.png`
- `visualizations/comparisons/heatmap_comparison.png`

### From main.py (Original Implementation)

```bash
python3 main.py
```

Generates:

- `visualizations/original_analysis/pca_variance_analysis.png`
- `visualizations/original_analysis/kmeans_elbow_curves.png`
- `visualizations/original_analysis/kmeans_original_2d.png`
- `visualizations/original_analysis/kmeans_pca_2d.png`
- `visualizations/comparisons/clustering_comparison.png`

## Visualization Details

### Experiment 1 Visualizations

**exp1_optimal_k_analysis.png** (4 panels):

1. **Elbow Curve**: Inertia vs k, helps identify optimal number of clusters
2. **Silhouette Analysis**: Silhouette scores for k=2 to 10
3. **Gap Statistic**: Gap values for determining optimal k
4. **Initialization Comparison**: K-Means++ vs Random (normalized metrics)

### Experiment 3 Visualizations

**exp3_pca_analysis.png** (6 panels):

1. **Variance Explained**: Cumulative variance by number of components
2. **Reconstruction Error**: MSE vs number of components
3. **Silhouette Scores**: Clustering quality vs components
4. **Purity Scores**: Purity vs components
5. **Davies-Bouldin Index**: Lower is better (cluster separation)
6. **Execution Time**: Computational efficiency vs components

### Comparison Visualizations

**confusion_matrices.png** (2 panels):

- Experiment 1: Confusion matrix for K-Means on original data
- Experiment 3: Confusion matrix for best PCA configuration

**clusters_2d_projections.png** (3 panels):

- True labels in 2D PCA projection
- Experiment 1 clusters (2D projection)
- Experiment 3 clusters (best configuration)

**heatmap_comparison.png**:

- All methods (Exp 1 K-Means++, Random, Exp 3 with 2/5/10/15/20 components)
- All metrics (Silhouette, Purity, Adjusted Rand, Normalized MI)
- Color-coded performance (green=good, red=poor)

### Original Analysis Visualizations

**pca_variance_analysis.png** (2 panels):

- Cumulative variance explained (with 90%/95% thresholds)
- Reconstruction error vs components

**kmeans_elbow_curves.png**:

- Elbow curves for original and PCA-reduced data side-by-side

**kmeans_original_2d.png**:

- 2D PCA projection with K-Means clusters on original data

**kmeans_pca_2d.png**:

- 2D visualization of clusters in PCA space

**clustering_comparison.png** (4 panels):

- True labels (2D projection)
- K-Means on original data (2D projection)
- K-Means on PCA data
- Cluster size comparison bar chart

## File Formats

- **Format**: PNG (Portable Network Graphics)
- **Resolution**: 300 DPI (high quality for reports)
- **Sizes**: Vary by complexity (138KB to 1.7MB)

## Usage in Reports

All visualizations are report-ready with:

- High resolution (300 DPI)
- Clear labels and titles
- Professional color schemes
- Legends and grid lines
- Proper axis labels

## Regenerating Visualizations

To regenerate all visualizations:

```bash
# Generate experiment visualizations
python3 main_updated.py

# Generate original analysis visualizations
python3 main.py
```

**Note**: Running the scripts will automatically create the folder structure if it doesn't exist.
