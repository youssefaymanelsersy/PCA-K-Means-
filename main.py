import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.datasets import load_breast_cancer
from pca import PCA
from kmeans import KMeans
from metrics import (silhouette_score, davies_bouldin_index, calinski_harabasz_index, adjusted_rand_index, normalized_mutual_information, purity, gap_statistic, confusion_matrix)

# Create visualization directories if they don't exist
os.makedirs('visualizations/experiment_1', exist_ok=True)
os.makedirs('visualizations/experiment_3', exist_ok=True)
os.makedirs('visualizations/original_analysis', exist_ok=True)
os.makedirs('visualizations/comparisons', exist_ok=True)


def load_data():    # Load Breast Cancer Wisconsin (Diagnostic) dataset
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))} (Malignant=0, Benign=1)")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names


def find_optimal_k_elbow(X, k_range, data_label="Data", random_state=42):   # Find optimal k using elbow method and silhouette analysis.
    print(f"\n  Elbow Method on {data_label}:")
    results = {'k': [], 'inertia': [], 'silhouette': [], 'time': []}
    
    for k in k_range:
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, max_iter=300, random_state=random_state)
        labels = kmeans.fit_predict(X)
        elapsed = time.time() - start_time
        
        sil = silhouette_score(X, labels)
        
        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(sil)
        results['time'].append(elapsed)
        
        print(f"    k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil:.4f}, time={elapsed:.3f}s")
    
    return results


def find_optimal_k_gap_statistic(X, k_range, n_refs=10, random_state=42):   # Find optimal k using gap statistic method.
    print(f"\n  Gap Statistic:")
    gaps = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, max_iter=300, random_state=random_state)
        labels = kmeans.fit_predict(X)
        gap = gap_statistic(X, labels, n_refs=n_refs, random_state=random_state)
        gaps.append(gap)
        print(f"    k={k}: gap={gap:.4f}")
    
    # Optimal k is where gap statistic is maximum
    optimal_idx = np.argmax(gaps)
    optimal_k = k_range[optimal_idx]
    print(f"  Optimal k (max gap): {optimal_k}")
    
    return gaps, optimal_k


def compute_comprehensive_metrics(X, labels, true_labels, execution_time=None):  # Compute a suite of clustering metrics.
    metrics = {}
    
    # Internal validation metrics
    metrics['silhouette'] = silhouette_score(X, labels)
    metrics['davies_bouldin'] = davies_bouldin_index(X, labels)
    metrics['calinski_harabasz'] = calinski_harabasz_index(X, labels)
    
    # WCSS (Within-cluster sum of squares)
    wcss = 0
    n_clusters = len(np.unique(labels))
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - center)**2)
    metrics['wcss'] = wcss
    
    # External validation metrics
    metrics['adjusted_rand'] = adjusted_rand_index(true_labels, labels)
    metrics['normalized_mutual_info'] = normalized_mutual_information(true_labels, labels)
    metrics['purity'] = purity(true_labels, labels)
    
    # Additional info
    metrics['n_clusters'] = n_clusters
    metrics['cluster_sizes'] = [np.sum(labels == i) for i in range(n_clusters)]
    
    if execution_time is not None:
        metrics['execution_time'] = execution_time
    
    return metrics


def experiment_1_kmeans_original(X, y, k_range=range(2, 11)):   # Experiment 1: K-Means on Original Data
    print("\n" + "="*80)
    print("EXPERIMENT 1: K-MEANS ON ORIGINAL DATA")
    print("="*80)
    
    results = {'elbow': {}, 'gap': {}, 'kmeans++': {}, 'random': {}}
    
    # 1. Elbow Method and Silhouette Analysis
    print("\n1. Finding Optimal k:")
    results['elbow'] = find_optimal_k_elbow(X, k_range, "Original Data")
    
    # 2. Gap Statistic
    gaps, optimal_k_gap = find_optimal_k_gap_statistic(X, k_range, n_refs=10)
    results['gap']['gaps'] = gaps
    results['gap']['optimal_k'] = optimal_k_gap
    
    # Determine optimal k (using silhouette score)
    optimal_idx = np.argmax(results['elbow']['silhouette'])
    optimal_k = results['elbow']['k'][optimal_idx]
    print(f"\n  Optimal k (silhouette): {optimal_k}")
    
    # 3. Compare K-Means++ vs Random Initialization
    print(f"\n2. K-Means++ vs Random Initialization (k={optimal_k}):")
    
    # K-Means++
    print("  K-Means++:")
    start_time = time.time()
    kmeans_pp = KMeans(n_clusters=optimal_k, init="kmeans++", max_iter=300, random_state=42)
    labels_pp = kmeans_pp.fit_predict(X)
    time_pp = time.time() - start_time
    metrics_pp = compute_comprehensive_metrics(X, labels_pp, y, time_pp)
    metrics_pp['n_iter'] = kmeans_pp.n_iter_
    metrics_pp['inertia'] = kmeans_pp.inertia_
    results['kmeans++'] = metrics_pp
    
    print(f"    Iterations: {kmeans_pp.n_iter_}")
    print(f"    Time: {time_pp:.4f}s")
    print(f"    Inertia: {kmeans_pp.inertia_:.2f}")
    print(f"    Silhouette: {metrics_pp['silhouette']:.4f}")
    print(f"    Purity: {metrics_pp['purity']:.4f}")
    
    # Random initialization
    print("  Random Initialization:")
    start_time = time.time()
    kmeans_rand = KMeans(n_clusters=optimal_k, init="random", max_iter=300, random_state=42)
    labels_rand = kmeans_rand.fit_predict(X)
    time_rand = time.time() - start_time
    metrics_rand = compute_comprehensive_metrics(X, labels_rand, y, time_rand)
    metrics_rand['n_iter'] = kmeans_rand.n_iter_
    metrics_rand['inertia'] = kmeans_rand.inertia_
    results['random'] = metrics_rand
    
    print(f"    Iterations: {kmeans_rand.n_iter_}")
    print(f"    Time: {time_rand:.4f}s")
    print(f"    Inertia: {kmeans_rand.inertia_:.2f}")
    print(f"    Silhouette: {metrics_rand['silhouette']:.4f}")
    print(f"    Purity: {metrics_rand['purity']:.4f}")
    
    # Convergence speed comparison
    print(f"\n  Convergence Speed Comparison:")
    print(f"    K-Means++ converged {(metrics_rand['n_iter'] - metrics_pp['n_iter'])} iterations faster")
    print(f"    K-Means++ is {(time_rand / time_pp):.2f}x faster")
    
    results['optimal_k'] = optimal_k
    results['labels_best'] = labels_pp  # Use K-Means++ as best
    
    return results


def experiment_3_kmeans_after_pca(X, y, n_components_list=[2, 5, 10, 15, 20], k=2):   # Experiment 3: K-Means after PCA
    print("\n" + "="*80)
    print("EXPERIMENT 3: K-MEANS AFTER PCA")
    print("="*80)
    
    results = {'n_components': [], 'pca_results': [], 'kmeans_results': [], 'metrics': [], 'reconstruction_errors': [], 'variance_explained': []}
    
    print(f"\nTesting PCA with components: {n_components_list}")
    print(f"K-Means with k={k}\n")
    
    for n_comp in n_components_list:
        print(f"{'='*60}")
        print(f"PCA with {n_comp} components:")
        print(f"{'='*60}")
        
        # Perform PCA
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        
        # PCA metrics
        cumulative_var = pca.get_cumulative_variance_ratio()[-1]
        print(f"  Variance explained: {cumulative_var:.4f}")
        
        # Reconstruction error
        X_reconstructed = pca.inverse_transform(X_pca)
        recon_error = np.mean((X - X_reconstructed)**2)
        print(f"  Reconstruction error (MSE): {recon_error:.4f}")
        
        # K-Means on PCA data
        print(f"  K-Means clustering:")
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        elapsed = time.time() - start_time
        
        print(f"    Iterations: {kmeans.n_iter_}")
        print(f"    Time: {elapsed:.4f}s")
        print(f"    Inertia: {kmeans.inertia_:.2f}")
        
        # Compute all metrics
        metrics = compute_comprehensive_metrics(X_pca, labels, y, elapsed)
        metrics['n_iter'] = kmeans.n_iter_
        metrics['inertia'] = kmeans.inertia_
        
        print(f"    Silhouette: {metrics['silhouette']:.4f}")
        print(f"    Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
        print(f"    Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
        print(f"    Adjusted Rand Index: {metrics['adjusted_rand']:.4f}")
        print(f"    Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
        print(f"    Purity: {metrics['purity']:.4f}")
        
        # Store results
        results['n_components'].append(n_comp)
        results['pca_results'].append({'pca': pca, 'X_pca': X_pca})
        results['kmeans_results'].append({'kmeans': kmeans, 'labels': labels})
        results['metrics'].append(metrics)
        results['reconstruction_errors'].append(recon_error)
        results['variance_explained'].append(cumulative_var)
    
    # Analysis
    print(f"\n{'='*80}")
    print("TRADE-OFF ANALYSIS")
    print(f"{'='*80}")
    
    # Find best by different criteria
    best_silhouette_idx = np.argmax([m['silhouette'] for m in results['metrics']])
    best_purity_idx = np.argmax([m['purity'] for m in results['metrics']])
    best_ari_idx = np.argmax([m['adjusted_rand'] for m in results['metrics']])
    
    print(f"Best by Silhouette: {n_components_list[best_silhouette_idx]} components")
    print(f"Best by Purity: {n_components_list[best_purity_idx]} components")
    print(f"Best by Adjusted Rand Index: {n_components_list[best_ari_idx]} components")
    
    return results


def plot_experiment_1_results(exp1_results, save_prefix="exp1"):
    """Plot results from Experiment 1."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Elbow curve
    ax = axes[0, 0]
    k_values = exp1_results['elbow']['k']
    inertias = exp1_results['elbow']['inertia']
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark optimal k
    optimal_k = exp1_results['optimal_k']
    optimal_idx = k_values.index(optimal_k)
    ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax.legend()
    
    # Silhouette scores
    ax = axes[0, 1]
    silhouettes = exp1_results['elbow']['silhouette']
    ax.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax.legend()
    
    # Gap statistic
    ax = axes[1, 0]
    gaps = exp1_results['gap']['gaps']
    ax.plot(k_values, gaps, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Gap Statistic', fontsize=12)
    ax.set_title('Gap Statistic Method', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    optimal_k_gap = exp1_results['gap']['optimal_k']
    ax.axvline(x=optimal_k_gap, color='darkred', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k_gap}')
    ax.legend()
    
    # K-Means++ vs Random comparison
    ax = axes[1, 1]
    methods = ['K-Means++', 'Random Init']
    metrics_comparison = {
        'Iterations': [exp1_results['kmeans++']['n_iter'], exp1_results['random']['n_iter']],
        'Time (s)': [exp1_results['kmeans++']['execution_time'], exp1_results['random']['execution_time']],
        'Silhouette': [exp1_results['kmeans++']['silhouette'], exp1_results['random']['silhouette']],
    }
    
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (metric_name, values) in enumerate(metrics_comparison.items()):
        # Normalize values for visualization
        norm_values = np.array(values) / np.max(values)
        ax.bar(x + i*width, norm_values, width, label=metric_name, alpha=0.8)
    
    ax.set_xlabel('Initialization Method', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('K-Means++ vs Random Init (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/experiment_1/{save_prefix}_optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: visualizations/experiment_1/{save_prefix}_optimal_k_analysis.png")
    plt.close()


def plot_experiment_3_results(exp3_results, save_prefix="exp3"):
    n_components = exp3_results['n_components']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Variance explained
    ax = axes[0, 0]
    ax.plot(n_components, exp3_results['variance_explained'], 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax.set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction error
    ax = axes[0, 1]
    ax.plot(n_components, exp3_results['reconstruction_errors'], 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_title('Reconstruction Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax = axes[0, 2]
    silhouettes = [m['silhouette'] for m in exp3_results['metrics']]
    ax.plot(n_components, silhouettes, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Clustering Quality (Silhouette)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Purity scores
    ax = axes[1, 0]
    purities = [m['purity'] for m in exp3_results['metrics']]
    ax.plot(n_components, purities, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Purity', fontsize=12)
    ax.set_title('Clustering Purity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Davies-Bouldin Index (lower is better)
    ax = axes[1, 1]
    db_scores = [m['davies_bouldin'] for m in exp3_results['metrics']]
    ax.plot(n_components, db_scores, 'co-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title('Davies-Bouldin Index (Lower=Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Execution time
    ax = axes[1, 2]
    times = [m['execution_time'] for m in exp3_results['metrics']]
    ax.plot(n_components, times, 'yo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Execution Time (s)', fontsize=12)
    ax.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/experiment_3/{save_prefix}_pca_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: visualizations/experiment_3/{save_prefix}_pca_analysis.png")
    plt.close()


def plot_confusion_matrices(exp1_results, exp3_results, y, save_prefix="confusion"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Experiment 1
    ax = axes[0]
    labels_exp1 = exp1_results['labels_best']
    cm1 = confusion_matrix(y, labels_exp1)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Cluster', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Experiment 1: K-Means on Original Data', fontsize=14, fontweight='bold')
    
    # Experiment 3 (best by silhouette)
    ax = axes[1]
    best_idx = np.argmax([m['silhouette'] for m in exp3_results['metrics']])
    labels_exp3 = exp3_results['kmeans_results'][best_idx]['labels']
    n_comp_best = exp3_results['n_components'][best_idx]
    cm2 = confusion_matrix(y, labels_exp3)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Cluster', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Experiment 3: K-Means after PCA ({n_comp_best} comp)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/comparisons/{save_prefix}_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Saved: visualizations/comparisons/{save_prefix}_matrices.png")
    plt.close()


def plot_2d_clusters(X, y, exp1_results, exp3_results, save_prefix="clusters"):
    # Create 2D PCA for visualization
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True labels
    ax = axes[0]
    for label in np.unique(y):
        mask = y == label
        label_name = 'Malignant' if label == 0 else 'Benign'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label_name, alpha=0.6, s=50)
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title('True Labels (2D PCA Projection)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Experiment 1 clusters
    ax = axes[1]
    labels_exp1 = exp1_results['labels_best']
    for cluster in np.unique(labels_exp1):
        mask = labels_exp1 == cluster
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'Cluster {cluster}', alpha=0.6, s=50)
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title('Exp 1: K-Means on Original Data', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Experiment 3 clusters (best)
    ax = axes[2]
    best_idx = np.argmax([m['silhouette'] for m in exp3_results['metrics']])
    labels_exp3 = exp3_results['kmeans_results'][best_idx]['labels']
    X_pca_best = exp3_results['pca_results'][best_idx]['X_pca']
    n_comp_best = exp3_results['n_components'][best_idx]
    
    # If more than 2 components, project to 2D for visualization
    if X_pca_best.shape[1] > 2:
        pca_vis = PCA(n_components=2)
        X_vis = pca_vis.fit_transform(X_pca_best)
    else:
        X_vis = X_pca_best
    
    for cluster in np.unique(labels_exp3):
        mask = labels_exp3 == cluster
        ax.scatter(X_vis[mask, 0], X_vis[mask, 1], label=f'Cluster {cluster}', alpha=0.6, s=50)
    ax.set_xlabel('First Component', fontsize=12)
    ax.set_ylabel('Second Component', fontsize=12)
    ax.set_title(f'Exp 3: K-Means after PCA ({n_comp_best} comp)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/comparisons/{save_prefix}_2d_projections.png', dpi=300, bbox_inches='tight')
    print(f"Saved: visualizations/comparisons/{save_prefix}_2d_projections.png")
    plt.close()


def create_comparison_table(exp1_results, exp3_results):
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS COMPARISON TABLE")
    print("="*80)
    
    # Experiment 1
    print("\nEXPERIMENT 1: K-MEANS ON ORIGINAL DATA")
    print("-" * 80)
    print(f"{'Metric':<30} {'K-Means++':<15} {'Random Init':<15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('Silhouette Score', 'silhouette', '{:.4f}'),
        ('Davies-Bouldin Index', 'davies_bouldin', '{:.4f}'),
        ('Calinski-Harabasz', 'calinski_harabasz', '{:.2f}'),
        ('Adjusted Rand Index', 'adjusted_rand', '{:.4f}'),
        ('Normalized Mutual Info', 'normalized_mutual_info', '{:.4f}'),
        ('Purity', 'purity', '{:.4f}'),
        ('WCSS', 'wcss', '{:.2f}'),
        ('Iterations', 'n_iter', '{:d}'),
        ('Time (s)', 'execution_time', '{:.4f}'),
    ]
    
    for metric_name, metric_key, fmt in metrics_to_compare:
        val_pp = exp1_results['kmeans++'][metric_key]
        val_rand = exp1_results['random'][metric_key]
        print(f"{metric_name:<30} {fmt.format(val_pp):<15} {fmt.format(val_rand):<15}")
    
    # Experiment 3
    print("\n\nEXPERIMENT 3: K-MEANS AFTER PCA")
    print("-" * 140)
    header = f"{'Comp':<6}"
    metric_names_short = ['Silhouette', 'Davies-B', 'Calinski-H', 'Adj.Rand', 'Norm.MI', 'Purity', 'WCSS', 'Iters', 'Time(s)']
    for name in metric_names_short:
        header += f"{name:>12}"
    print(header)
    print("-" * 140)
    
    for i, n_comp in enumerate(exp3_results['n_components']):
        row = f"{n_comp:<6}"
        for metric_name, metric_key, fmt in metrics_to_compare:
            val = exp3_results['metrics'][i][metric_key]
            if 'f' in fmt:
                # Adjust formatting width
                if metric_key in ['calinski_harabasz', 'wcss']:
                    row += f"{val:>12.1f}"
                else:
                    row += f"{val:>12.4f}"
            else:
                row += f"{val:>12d}"
        print(row)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    # Best from Experiment 1
    if exp1_results['kmeans++']['purity'] > exp1_results['random']['purity']:
        print("\nExperiment 1: K-Means++ outperforms Random Initialization")
        print(f"  - Better purity: {exp1_results['kmeans++']['purity']:.4f} vs {exp1_results['random']['purity']:.4f}")
        print(f"  - Faster convergence: {exp1_results['kmeans++']['n_iter']} vs {exp1_results['random']['n_iter']} iterations")
    
    # Best from Experiment 3
    best_idx = np.argmax([m['purity'] for m in exp3_results['metrics']])
    best_n_comp = exp3_results['n_components'][best_idx]
    best_metrics = exp3_results['metrics'][best_idx]
    
    print(f"\nExperiment 3: Best performance with {best_n_comp} components")
    print(f"  - Purity: {best_metrics['purity']:.4f}")
    print(f"  - Silhouette: {best_metrics['silhouette']:.4f}")
    print(f"  - Variance explained: {exp3_results['variance_explained'][best_idx]:.4f}")
    print(f"  - Dimensionality reduction: {30} → {best_n_comp} features ({100*(1-best_n_comp/30):.1f}% reduction)")
    
    # Overall comparison
    print("\nOverall Comparison:")
    exp1_purity = exp1_results['kmeans++']['purity']
    exp3_purity = best_metrics['purity']
    
    if exp3_purity > exp1_purity:
        improvement = (exp3_purity - exp1_purity) / exp1_purity * 100
        print(f"  PCA preprocessing improves clustering quality by {improvement:.2f}%")
    else:
        print(f"  Original data performs slightly better, but PCA offers computational benefits")


def plot_metrics_heatmap(exp1_results, exp3_results, save_prefix="heatmap"):
    """Create heatmap comparing all methods across all metrics."""
    # Prepare data
    methods = ['Exp1: K-Means++', 'Exp1: Random'] + \
    [f'Exp3: PCA({n})' for n in exp3_results['n_components']]
    
    metrics_names = ['Silhouette', 'Purity', 'Adj. Rand', 'Norm. MI']
    metrics_keys = ['silhouette', 'purity', 'adjusted_rand', 'normalized_mutual_info']
    
    data = []
    
    # Exp 1
    data.append([exp1_results['kmeans++'][k] for k in metrics_keys])
    data.append([exp1_results['random'][k] for k in metrics_keys])
    
    # Exp 3
    for metrics in exp3_results['metrics']:
        data.append([metrics[k] for k in metrics_keys])
    
    data = np.array(data)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=methods, yticklabels=metrics_names,
                cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    plt.xlabel('Methods', fontsize=12)
    plt.ylabel('Metrics', fontsize=12)
    plt.title('Performance Comparison Heatmap\n(Higher is Better)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'visualizations/comparisons/{save_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: visualizations/comparisons/{save_prefix}_comparison.png")
    plt.close()


def main():
    print("\n" + "="*80)
    print("ASSIGNMENT 4: PCA & K-MEANS CLUSTERING ANALYSIS")
    print("Breast Cancer Wisconsin Dataset")
    print("Experiments 1 & 3")
    print("="*80)
    
    # Load data
    X, y, feature_names = load_data()
    
    # Experiment 1: K-Means on original data
    exp1_results = experiment_1_kmeans_original(X, y, k_range=range(2, 11))
    
    # Experiment 3: K-Means after PCA
    k_optimal = exp1_results['optimal_k']
    exp3_results = experiment_3_kmeans_after_pca(X, y, n_components_list=[2, 5, 10, 15, 20], k=k_optimal)
    
    # Generate all visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_experiment_1_results(exp1_results)
    plot_experiment_3_results(exp3_results)
    plot_confusion_matrices(exp1_results, exp3_results, y)
    plot_2d_clusters(X, y, exp1_results, exp3_results)
    plot_metrics_heatmap(exp1_results, exp3_results)
    
    # Create comparison table
    create_comparison_table(exp1_results, exp3_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated visualizations organized in folders:")
    print("  visualizations/experiment_1/")
    print("    - exp1_optimal_k_analysis.png")
    print("  visualizations/experiment_3/")
    print("    - exp3_pca_analysis.png")
    print("  visualizations/comparisons/")
    print("    - confusion_matrices.png")
    print("    - clusters_2d_projections.png")
    print("    - heatmap_comparison.png")
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
