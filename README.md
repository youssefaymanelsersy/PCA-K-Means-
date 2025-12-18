# PCA and K-Means Clustering Analysis

This project implements **PCA (Principal Component Analysis)** and **K-Means Clustering** from scratch using only NumPy for unsupervised learning on the **Breast Cancer Wisconsin dataset**.

## 🎯 Project Overview

This implementation follows the Assignment 4 specification and includes:

- **PCA Implementation from Scratch**
  - Eigenvalue-based decomposition
  - Data standardization (zero mean, unit variance)
  - Covariance matrix computation
  - Dimensionality reduction
  - Data reconstruction capability
  
- **K-Means Clustering from Scratch**
  - K-Means++ initialization for better convergence
  - Iterative cluster assignment and centroid updates
  - Convergence tracking
  - Inertia (WCSS) calculation
  
- **Comprehensive Analysis**
  - Dimensionality experiments (2-20 components)
  - K-Means on original data
  - K-Means on PCA-reduced data
  - Elbow curve analysis
  - Clustering evaluation (purity, inertia)
  - Multiple visualizations

## 📋 Requirements

- Python 3.7+
- NumPy (for core implementations)
- Matplotlib (for visualizations)
- scikit-learn (only for loading the dataset)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/youssefaymanelsersy/PCA_-_K-Means_Clusters.git
cd PCA_-_K-Means_Clusters

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Usage

Run the main analysis script:

```bash
python main_updated.py
```

This will:

1. Load the Breast Cancer Wisconsin dataset
2. Perform PCA experiments with 2-20 components
3. Run K-Means clustering on both original and PCA-reduced data
4. Generate comprehensive visualizations under `visualizations/`
5. Output detailed analysis results

## 📊 Generated Visualizations

The script generates the following plots:

1. **pca_variance_analysis.png**
   - Cumulative variance explained by components
   - Reconstruction error vs number of components

2. **kmeans_elbow_curves.png**
   - Elbow curves for determining optimal k
   - Comparison of original vs PCA-reduced data

3. **kmeans_original_2d.png**
   - 2D visualization of clusters on original data

4. **kmeans_pca_2d.png**
   - 2D visualization of clusters on PCA-reduced data

5. **clustering_comparison.png**
   - Comprehensive 4-panel comparison showing:
     - True labels
     - K-Means on original data
     - K-Means on PCA data
     - Cluster size comparison

## 🔬 Implementation Details

### PCA (pca.py)

The PCA implementation uses eigenvalue decomposition:

```python
from pca import PCA

# Create PCA with n components
pca = PCA(n_components=10)

# Fit and transform data
X_transformed = pca.fit_transform(X)

# Reconstruct data
X_reconstructed = pca.inverse_transform(X_transformed)

# Get explained variance
variance_ratio = pca.explained_variance_ratio_
```

**Key Features:**

- Data standardization (z-score normalization)
- Covariance matrix computation
- Eigenvalue/eigenvector calculation
- Component selection based on explained variance
- Reconstruction back to original space

### K-Means (kmeans.py)

The K-Means implementation uses K-Means++ initialization:

```python
from kmeans import KMeans

# Create K-Means with k clusters
kmeans = KMeans(n_clusters=2, max_iter=300, random_state=42)

# Fit and predict
labels = kmeans.fit_predict(X)

# Access results
print(f"Inertia: {kmeans.inertia_}")
print(f"Iterations: {kmeans.n_iter_}")
print(f"Inertia history: {kmeans.inertia_history_}")
```

**Key Features:**

- K-Means++ initialization for better initial centroids
- Convergence tracking with tolerance
- Inertia calculation (within-cluster sum of squares)
- Iteration history for analysis
- Handles empty clusters gracefully

## 📈 Analysis Workflow

1. **Data Loading**: Load Breast Cancer Wisconsin dataset (569 samples, 30 features)

2. **PCA Experiments**: Test with 2-20 components
   - Calculate explained variance ratios
   - Compute reconstruction errors
   - Identify optimal components (95% variance threshold)

3. **K-Means Analysis**:

   - Perform elbow analysis (k=2 to 10)
   - Compare original vs PCA-reduced data
   - Final clustering with optimal k (k=2 for binary classification)

4. **Evaluation**:
   - Calculate purity scores
   - Compare inertia values
   - Analyze convergence behavior
   - Visualize cluster distributions

## 🎓 Educational Value

This project demonstrates:

- **Eigenvalue decomposition** for dimensionality reduction
- **K-Means++ initialization** for improved clustering
- **Convergence criteria** in iterative algorithms
- **Elbow method** for selecting optimal k
- **PCA benefits** for high-dimensional data clustering
- **Unsupervised learning** evaluation techniques

## 📝 Key Results

The analysis typically shows:

- **PCA**: ~10-12 components capture 95% of variance
- **K-Means Convergence**: Usually converges in 5-20 iterations
- **Clustering Purity**: Typically 85-95% agreement with true labels
- **PCA Benefit**: Reduced computational cost with similar or better clustering quality

## 🔧 Code Structure

```english
PCA_-_K-Means_Clusters/
│
├── pca.py               # PCA implementation from scratch
├── kmeans.py            # K-Means implementation from scratch
├── metrics.py           # Metrics implemented from scratch
├── main_updated.py      # Main analysis script (Experiments 1 & 3)
├── test_implementation.py
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── visualizations/      # Organized output plots
```

## 🤝 Contributing

This is an educational project. Feel free to:

- Report issues
- Suggest improvements
- Fork and experiment

## 📄 License

This project is for educational purposes as part of Assignment 4.

## 👤 Author

Youssef Ayman Elsersy

## 🙏 Acknowledgments

- Breast Cancer Wisconsin dataset from UCI Machine Learning Repository
- Assignment 4 specification for project requirements
