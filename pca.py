import numpy as np


class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.std_ = None
        self.eigenvalues_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        # Avoid division by zero for constant features
        # Set std to 1 for features with zero variance (no variation to standardize)
        self.std_[self.std_ == 0] = 1
        
        # Standardize the data
        X_std = (X - self.mean_) / self.std_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_std.T)
        
        # Compute eigenvalues and eigenvectors using eigh for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store all eigenvalues for variance calculation
        self.eigenvalues_ = eigenvalues
        
        # Select top n_components
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        # Standardize the data using training statistics
        X_std = (X - self.mean_) / self.std_
        
        # Project onto principal components
        X_transformed = np.dot(X_std, self.components_)
        
        return X_transformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        # Project back to original space
        X_std_reconstructed = np.dot(X_transformed, self.components_.T)
        
        # Reverse standardization
        X_reconstructed = X_std_reconstructed * self.std_ + self.mean_
        
        return X_reconstructed
    
    def get_cumulative_variance_ratio(self):
        return np.cumsum(self.explained_variance_ratio_)
