import numpy as np
class KMeans:

    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4,init="kmeans++", random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.inertia_history_ = []
        
    def _random_init(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
        
    def _kmeans_plusplus_init(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Choose first center randomly
        centers = [X[np.random.randint(n_samples)]]
        
        # Choose remaining centers
        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest center for each point
            # Using vectorized operations for efficiency
            centers_array = np.array(centers)
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centers_array, axis=2)**2, axis=1)
            
            # Choose next center with probability proportional to distance squared
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            for idx, cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    centers.append(X[idx])
                    break
                    
        return np.array(centers)
    
    def _assign_clusters(self, X, centers):
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
            
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centers(self, X, labels):
        centers = np.zeros((self.n_clusters, X.shape[1]))
        
        # Use RandomState for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, reinitialize randomly with reproducibility
                centers[i] = X[rng.randint(X.shape[0])]
                
        return centers
    
    def _calculate_inertia(self, X, labels, centers):
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[i])**2)
        return inertia
    
    def fit(self, X):
        # Initialize centers using specified method
        if self.init == "random":
            self.cluster_centers_ = self._random_init(X)
        else:
            self.cluster_centers_ = self._kmeans_plusplus_init(X)

        self.inertia_history_ = []
        
        # Iterate until convergence or max_iter
        for iteration in range(self.max_iter):
            # Assign clusters
            old_centers = self.cluster_centers_.copy()
            self.labels_ = self._assign_clusters(X, self.cluster_centers_)
            
            # Update centers
            self.cluster_centers_ = self._update_centers(X, self.labels_)
            
            # Calculate inertia
            inertia = self._calculate_inertia(X, self.labels_, self.cluster_centers_)
            self.inertia_history_.append(inertia)
            
            # Check convergence
            center_shift = np.linalg.norm(self.cluster_centers_ - old_centers)
            if center_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
            
        # Store final inertia
        self.inertia_ = self.inertia_history_[-1] if self.inertia_history_ else 0
        
        return self
    
    def predict(self, X):
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
