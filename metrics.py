import numpy as np
from scipy.special import comb


def silhouette_score(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    silhouettes = []

    for i in range(n_samples):
        cluster_i = labels[i]
        same_cluster = X[labels == cluster_i]
        
        # Compute a(i) - mean distance to points in same cluster
        if len(same_cluster) > 1:
            a_i = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a_i = 0
        
        # Compute b(i) - mean distance to points in nearest other cluster
        b_i = np.inf
        for cluster_j in unique_labels: 
            if cluster_j != cluster_i:
                other_cluster = X[labels == cluster_j]
                mean_dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
                b_i = min(b_i, mean_dist)
        
        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0
        silhouettes.append(s_i)

    return np.mean(silhouettes)


def davies_bouldin_index(X, labels):
    n_clusters = len(np.unique(labels))
    
    # Compute cluster centers
    centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    
    # Compute average within-cluster distances
    S = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            S[i] = np.mean(np.linalg.norm(cluster_points - centers[i], axis=1))
    
    # Compute Davies-Bouldin Index
    db_values = []
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                center_dist = np.linalg.norm(centers[i] - centers[j])
                if center_dist > 0:
                    ratio = (S[i] + S[j]) / center_dist
                    max_ratio = max(max_ratio, ratio)
        db_values.append(max_ratio)
    
    return np.mean(db_values)


def calinski_harabasz_index(X, labels):
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Overall mean
    overall_mean = X.mean(axis=0)
    
    # Between-cluster dispersion
    cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
    cluster_sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
    
    between_dispersion = np.sum(cluster_sizes * np.sum((cluster_centers - overall_mean)**2, axis=1))
    
    # Within-cluster dispersion
    within_dispersion = 0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        within_dispersion += np.sum((cluster_points - cluster_centers[i])**2)
    
    if within_dispersion == 0:
        return 0.0
    
    # Calinski-Harabasz score
    score = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
    
    return score


def adjusted_rand_index(true_labels, pred_labels):
    n = len(true_labels)
    
    # Create contingency table
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    contingency = np.zeros((len(true_unique), len(pred_unique)))
    for i, true_val in enumerate(true_unique):
        for j, pred_val in enumerate(pred_unique):
            contingency[i, j] = np.sum((true_labels == true_val) & (pred_labels == pred_val))
    
    # Sum of combinations for each cell
    sum_comb_c = np.sum([comb(n_ij, 2) for n_ij in contingency.flatten() if n_ij >= 2])
    
    # Sum of combinations for rows and columns
    sum_comb_rows = np.sum([comb(n_i, 2) for n_i in np.sum(contingency, axis=1) if n_i >= 2])
    sum_comb_cols = np.sum([comb(n_j, 2) for n_j in np.sum(contingency, axis=0) if n_j >= 2])
    
    # Expected index
    expected_index = sum_comb_rows * sum_comb_cols / comb(n, 2)
    
    # Max index
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    
    # Adjusted Rand Index
    if max_index - expected_index == 0:
        return 0.0
    
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    
    return ari


def normalized_mutual_information(true_labels, pred_labels):
    n = len(true_labels)
    
    # Get unique labels
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    # Create contingency table
    contingency = np.zeros((len(true_unique), len(pred_unique)))
    for i, true_val in enumerate(true_unique):
        for j, pred_val in enumerate(pred_unique):
            contingency[i, j] = np.sum((true_labels == true_val) & (pred_labels == pred_val))
    
    # Compute marginals
    true_marginal = np.sum(contingency, axis=1)
    pred_marginal = np.sum(contingency, axis=0)
    
    # Compute mutual information
    mi = 0.0
    for i in range(len(true_unique)):
        for j in range(len(pred_unique)):
            if contingency[i, j] > 0:
                mi += contingency[i, j] * np.log((n * contingency[i, j]) / (true_marginal[i] * pred_marginal[j]))
    mi /= n
    
    # Compute entropies
    h_true = 0.0
    for count in true_marginal:
        if count > 0:
            h_true -= (count / n) * np.log(count / n)
    
    h_pred = 0.0
    for count in pred_marginal:
        if count > 0:
            h_pred -= (count / n) * np.log(count / n)
    
    # Normalized Mutual Information
    if h_true == 0 or h_pred == 0:
        return 0.0
    
    nmi = 2 * mi / (h_true + h_pred)
    
    return nmi


def purity(true_labels, pred_labels):
    n_clusters = len(np.unique(pred_labels))
    total_correct = 0
    
    for i in range(n_clusters):
        cluster_mask = (pred_labels == i)
        if np.sum(cluster_mask) > 0:
            # Find most common true label in this cluster
            true_labels_in_cluster = true_labels[cluster_mask]
            unique_labels, counts = np.unique(true_labels_in_cluster, return_counts=True)
            max_count = np.max(counts)
            total_correct += max_count
    
    purity_score = total_correct / len(true_labels)
    return purity_score


def gap_statistic(X, labels, n_refs=10, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Compute within-cluster sum of squares (WCSS) for actual data
    n_clusters = len(np.unique(labels))
    wcss = 0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            wcss += np.sum((cluster_points - center)**2)
    
    # Generate reference datasets and compute their WCSS
    ref_wcss = []
    for _ in range(n_refs):
        # Generate random data within the same bounds as X
        ref_data = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
        
        # Cluster the reference data with same k
        from kmeans import KMeans
        kmeans_ref = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_ref.fit(ref_data)
        ref_wcss.append(kmeans_ref.inertia_)
    
    # Compute gap statistic
    gap = np.log(np.mean(ref_wcss)) - np.log(wcss)
    
    return gap


def confusion_matrix(true_labels, pred_labels):
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    matrix = np.zeros((len(true_unique), len(pred_unique)), dtype=int)
    
    for i, true_val in enumerate(true_unique):
        for j, pred_val in enumerate(pred_unique):
            matrix[i, j] = np.sum((true_labels == true_val) & (pred_labels == pred_val))
    
    return matrix
