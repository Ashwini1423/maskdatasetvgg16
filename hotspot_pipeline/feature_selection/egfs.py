"""
Enhanced Graph-based Feature Selection (EGFS)
==============================================
Step 3: Intelligent Feature Selection (~3ms target)
- EGFS algorithm selects optimal descriptor combination
- Skip unnecessary computations in uniform regions
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import cv2


class EGFSSelector:
    """
    Enhanced Graph-based Feature Selection for optimal descriptor combination.
    Optimized for real-time performance with uniform region detection.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        uniform_threshold: float = 0.1,
        variance_threshold: float = 0.01
    ):
        """
        Initialize EGFS selector.

        Args:
            n_features: Number of top features to select (None = auto-select)
            uniform_threshold: Threshold for uniform region detection
            variance_threshold: Minimum variance for feature relevance
        """
        self.n_features = n_features
        self.uniform_threshold = uniform_threshold
        self.variance_threshold = variance_threshold

        self.selected_indices_ = None
        self.feature_scores_ = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def is_uniform_region(self, image: np.ndarray) -> bool:
        """
        Detect if image region is uniform (skip unnecessary computations).

        Args:
            image: Input image region

        Returns:
            True if region is uniform, False otherwise
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute local variance
        mean = np.mean(image)
        variance = np.var(image)
        normalized_variance = variance / (mean + 1e-7)

        return normalized_variance < self.uniform_threshold

    def compute_mutual_information(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute mutual information between features and labels.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples,)

        Returns:
            Mutual information scores for each feature
        """
        mi_scores = mutual_info_classif(features, labels, random_state=42)
        return mi_scores

    def compute_feature_redundancy(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise feature redundancy using correlation.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Redundancy matrix
        """
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(features.T)

        # Redundancy is average absolute correlation with other features
        redundancy = np.mean(np.abs(correlation_matrix), axis=1)

        return redundancy

    def compute_graph_score(
        self,
        relevance: np.ndarray,
        redundancy: np.ndarray,
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Compute graph-based feature scores.

        Args:
            relevance: Feature relevance scores
            redundancy: Feature redundancy scores
            alpha: Weight for relevance vs redundancy trade-off

        Returns:
            Combined feature scores
        """
        # Normalize scores
        relevance_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-7)
        redundancy_norm = (redundancy - redundancy.min()) / (redundancy.max() - redundancy.min() + 1e-7)

        # Graph-based score: maximize relevance, minimize redundancy
        scores = alpha * relevance_norm - (1 - alpha) * redundancy_norm

        return scores

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.7
    ) -> 'EGFSSelector':
        """
        Fit EGFS selector on training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            alpha: Relevance vs redundancy trade-off

        Returns:
            Self
        """
        # Filter low-variance features
        variances = np.var(X, axis=0)
        valid_indices = variances > self.variance_threshold

        if not np.any(valid_indices):
            raise ValueError("No features with sufficient variance")

        X_filtered = X[:, valid_indices]

        # Compute feature relevance (mutual information)
        relevance = self.compute_mutual_information(X_filtered, y)

        # Compute feature redundancy
        redundancy = self.compute_feature_redundancy(X_filtered)

        # Compute graph-based scores
        graph_scores = self.compute_graph_score(relevance, redundancy, alpha)

        # Select top features
        if self.n_features is None:
            # Auto-select features above mean score
            threshold = np.mean(graph_scores)
            selected_mask = graph_scores > threshold
            self.selected_indices_ = np.where(valid_indices)[0][selected_mask]
        else:
            # Select top-k features
            n_features = min(self.n_features, len(graph_scores))
            top_indices = np.argsort(graph_scores)[-n_features:]
            self.selected_indices_ = np.where(valid_indices)[0][top_indices]

        # Store full scores for all features
        self.feature_scores_ = np.zeros(X.shape[1])
        self.feature_scores_[valid_indices] = graph_scores

        # Fit scaler on selected features
        X_selected = X[:, self.selected_indices_]
        self.scaler.fit(X_selected)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features by selecting optimal subset.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Selected and scaled features
        """
        if not self.is_fitted:
            raise ValueError("EGFSSelector must be fitted before transform")

        # Select features
        X_selected = X[:, self.selected_indices_]

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        return X_scaled

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Training features
            y: Training labels
            alpha: Relevance vs redundancy trade-off

        Returns:
            Selected and scaled features
        """
        self.fit(X, y, alpha)
        return self.transform(X)

    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if not self.is_fitted:
            raise ValueError("EGFSSelector must be fitted first")

        importance = {}
        for idx in range(len(self.feature_scores_)):
            importance[idx] = self.feature_scores_[idx]

        return importance

    def get_selected_features(self) -> np.ndarray:
        """
        Get indices of selected features.

        Returns:
            Array of selected feature indices
        """
        if not self.is_fitted:
            raise ValueError("EGFSSelector must be fitted first")

        return self.selected_indices_

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Allow selector to be called as a function."""
        return self.transform(X)


class AdaptiveFeatureSelector:
    """
    Adaptive feature selector that skips computation in uniform regions.
    Wrapper around EGFS for real-time optimization.
    """

    def __init__(self, egfs_selector: EGFSSelector):
        """
        Initialize adaptive selector.

        Args:
            egfs_selector: Fitted EGFS selector
        """
        self.egfs_selector = egfs_selector

    def select_features(
        self,
        features: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select features with optional uniform region detection.

        Args:
            features: Input feature vector
            image: Optional original image for uniform detection

        Returns:
            Selected features
        """
        # Check if region is uniform (skip expensive computation)
        if image is not None and self.egfs_selector.is_uniform_region(image):
            # Return zero features for uniform regions (fast path)
            n_selected = len(self.egfs_selector.selected_indices_)
            return np.zeros(n_selected)

        # Normal feature selection path
        if features.ndim == 1:
            features = features.reshape(1, -1)

        selected = self.egfs_selector.transform(features)
        return selected.flatten()
