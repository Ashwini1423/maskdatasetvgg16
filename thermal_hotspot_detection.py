"""
Real-Time Thermal Hotspot Detection System
Based on: Real-Time Analysis of Thermal Hotspot Detection Using
Multi-Feature Extraction and Enhanced Graph-Based Selection

This module implements the complete pipeline for thermal hotspot detection
with real-time processing capabilities for VS Code execution.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)
from sklearn.model_selection import cross_val_score
from scipy.ndimage import uniform_filter
import time
import os

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION MODULES
# ============================================================================

class FeatureExtractor:
    """Base class for feature extraction methods."""

    @staticmethod
    def extract_lbp(image, radius=1, n_points=8):
        """
        Local Binary Pattern (LBP) feature extraction.

        Args:
            image: Input grayscale image
            radius: Radius of the circular pattern
            n_points: Number of neighborhood points

        Returns:
            LBP histogram and LBP map
        """
        h, w = image.shape
        lbp_map = np.zeros((h, w), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_string = ''

                # Circular neighborhood
                angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
                for angle in angles:
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)

                    # Bilinear interpolation
                    x1, y1 = int(np.floor(x)), int(np.floor(y))
                    x2, y2 = x1 + 1, y1 + 1

                    dx = x - x1
                    dy = y - y1

                    p1 = image[i + y1, j + x1] if 0 <= i+y1 < h and 0 <= j+x1 < w else center
                    p2 = image[i + y1, j + x2] if 0 <= i+y1 < h and 0 <= j+x2 < w else center
                    p3 = image[i + y2, j + x1] if 0 <= i+y2 < h and 0 <= j+x1 < w else center
                    p4 = image[i + y2, j + x2] if 0 <= i+y2 < h and 0 <= j+x2 < w else center

                    value = (1-dx)*(1-dy)*p1 + dx*(1-dy)*p2 + (1-dx)*dy*p3 + dx*dy*p4
                    binary_string += '1' if value >= center else '0'

                lbp_map[i, j] = int(binary_string, 2)

        # Compute histogram
        hist = np.histogram(lbp_map, bins=2**n_points, range=(0, 2**n_points))[0]
        hist = hist.astype(float) / (h * w)

        return hist, lbp_map

    @staticmethod
    def extract_loop(image, radius=1, n_points=8):
        """
        Local Oriented Pattern (LOOP) feature extraction.
        Extends LBP with orientation information.

        Args:
            image: Input grayscale image
            radius: Radius of the circular pattern
            n_points: Number of neighborhood points

        Returns:
            LOOP histogram and LOOP map
        """
        h, w = image.shape
        loop_map = np.zeros((h, w), dtype=np.uint8)

        # Compute gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                center_angle = np.arctan2(gy[i, j], gx[i, j])
                binary_string = ''

                angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
                for idx, angle in enumerate(angles):
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)

                    x1, y1 = int(np.floor(x)), int(np.floor(y))
                    x2, y2 = x1 + 1, y1 + 1
                    dx = x - x1
                    dy = y - y1

                    p1 = image[i + y1, j + x1] if 0 <= i+y1 < h and 0 <= j+x1 < w else center
                    p2 = image[i + y1, j + x2] if 0 <= i+y1 < h and 0 <= j+x2 < w else center
                    p3 = image[i + y2, j + x1] if 0 <= i+y2 < h and 0 <= j+x1 < w else center
                    p4 = image[i + y2, j + x2] if 0 <= i+y2 < h and 0 <= j+x2 < w else center

                    value = (1-dx)*(1-dy)*p1 + dx*(1-dy)*p2 + (1-dx)*dy*p3 + dx*dy*p4

                    # Orientation-aware comparison
                    neighbor_angle = np.arctan2(gy[i + int(y), j + int(x)],
                                               gx[i + int(y), j + int(x)])
                    angle_diff = abs(neighbor_angle - center_angle)

                    binary_string += '1' if (value >= center) and (angle_diff < np.pi/4) else '0'

                loop_map[i, j] = int(binary_string, 2)

        hist = np.histogram(loop_map, bins=2**n_points, range=(0, 2**n_points))[0]
        hist = hist.astype(float) / (h * w)

        return hist, loop_map

    @staticmethod
    def extract_ldp(image, radius=1, n_points=8):
        """
        Local Directional Pattern (LDP) feature extraction.
        Captures directional information in edge responses.

        Args:
            image: Input grayscale image
            radius: Radius of the pattern
            n_points: Number of neighborhood points

        Returns:
            LDP histogram and LDP map
        """
        h, w = image.shape
        ldp_map = np.zeros((h, w), dtype=np.uint8)

        # Compute edge responses in multiple directions
        directions = 8
        edge_responses = []

        for d in range(directions):
            angle = (d * np.pi) / directions
            # Directional derivative filters
            k = np.array([[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]], dtype=np.float32)
            edge = cv2.filter2D(image, cv2.CV_32F, k)
            edge_responses.append(edge)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                binary_string = ''
                center_responses = [e[i, j] for e in edge_responses]

                for d in range(directions):
                    max_edge = np.max(edge_responses[d][i-radius:i+radius+1, j-radius:j+radius+1])
                    binary_string += '1' if center_responses[d] > max_edge * 0.5 else '0'

                ldp_map[i, j] = int(binary_string, 2)

        hist = np.histogram(ldp_map, bins=2**n_points, range=(0, 2**n_points))[0]
        hist = hist.astype(float) / (h * w)

        return hist, ldp_map

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_size=8):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Improves local contrast while preventing noise amplification.

        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid tiles

        Returns:
            CLAHE enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(image.astype(np.uint8))
        return enhanced.astype(np.float32)

    @staticmethod
    def multi_resolution_analysis(image, scales=[1, 2, 4]):
        """
        Multi-resolution feature extraction at different scales.

        Args:
            image: Input image
            scales: List of scale factors

        Returns:
            Multi-scale features
        """
        multi_features = []
        for scale in scales:
            scaled = cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale))
            # Extract features from scaled version
            lbp_hist, _ = FeatureExtractor.extract_lbp(scaled)
            multi_features.append(lbp_hist)

        return np.concatenate(multi_features)


# ============================================================================
# ENHANCED GRAPH-BASED FEATURE SELECTION (EGFS)
# ============================================================================

class EGFSSelector:
    """Enhanced Graph-Based Feature Selection for hotspot detection."""

    def __init__(self, n_features=50, weighting='laplacian'):
        """
        Initialize EGFS selector.

        Args:
            n_features: Number of features to select
            weighting: Type of graph weighting ('laplacian', 'knn', 'rbf')
        """
        self.n_features = n_features
        self.weighting = weighting
        self.selected_features = None
        self.scores = None
        self.feature_graph = None

    def _build_graph(self, X, k=10):
        """
        Build graph structure from feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features)
            k: Number of nearest neighbors

        Returns:
            Adjacency matrix and weight matrix
        """
        n_samples, n_features = X.shape

        # Compute feature-to-feature distances based on correlation/covariance
        feature_matrix = X.T  # Shape: (n_features, n_samples)
        from scipy.spatial.distance import cdist
        feature_distances = cdist(feature_matrix, feature_matrix, metric='correlation')
        feature_distances = np.nan_to_num(feature_distances, nan=1.0)  # Handle NaN from correlation

        # Build adjacency based on k-nearest neighbors among features
        adjacency = np.zeros((n_features, n_features))

        for i in range(n_features):
            nearest_indices = np.argsort(feature_distances[i])[:min(k+1, n_features)]
            adjacency[i, nearest_indices] = 1
            adjacency[nearest_indices, i] = 1

        np.fill_diagonal(adjacency, 0)  # Remove self-connections

        # Compute weight matrix based on feature importance
        if self.weighting == 'laplacian':
            degrees = np.sum(adjacency, axis=1)
            degrees = np.maximum(degrees, 1)  # Avoid division by zero
            laplacian = np.diag(degrees) - adjacency
            weights = np.linalg.pinv(laplacian + np.eye(n_features) * 0.1)
        elif self.weighting == 'rbf':
            gamma = 1.0 / (np.mean(feature_distances[feature_distances > 0]) + 1e-6)
            weights = np.exp(-gamma * feature_distances)
        else:  # knn
            weights = adjacency

        return adjacency, weights

    def _score_features(self, X, y, weights):
        """
        Score features using mutual information and graph weights.

        Args:
            X: Feature matrix
            y: Labels
            weights: Graph weights

        Returns:
            Feature importance scores
        """
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Combine mutual information with graph weights
        for i in range(n_features):
            graph_score = np.mean(weights[i])
            scores[i] = mi_scores[i] + 0.3 * graph_score

        return scores

    def fit(self, X, y):
        """
        Fit EGFS selector.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels
        """
        adjacency, weights = self._build_graph(X)
        self.feature_graph = adjacency

        scores = self._score_features(X, y, weights)
        self.scores = scores

        # Select top features
        top_indices = np.argsort(scores)[::-1][:self.n_features]
        self.selected_features = sorted(top_indices)

        return self

    def transform(self, X):
        """
        Transform features using selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.selected_features is None:
            raise ValueError("EGFS not fitted yet!")
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


# ============================================================================
# THERMAL HOTSPOT DETECTION PIPELINE
# ============================================================================

class ThermalHotspotDetector:
    """Complete thermal hotspot detection pipeline."""

    def __init__(self, clahe_clip=2.0, svm_kernel='rbf', egfs_features=50):
        """
        Initialize detector.

        Args:
            clahe_clip: CLAHE clip limit
            svm_kernel: SVM kernel type
            egfs_features: Number of features after EGFS selection
        """
        self.clahe_clip = clahe_clip
        self.svm_kernel = svm_kernel
        self.egfs_features = egfs_features

        self.egfs_selector = EGFSSelector(n_features=egfs_features)
        self.scaler = StandardScaler()
        self.svm_classifier = None

        self.feature_extractor = FeatureExtractor()
        self.training_history = {}

    def preprocess_image(self, image):
        """
        Preprocess thermal image.

        Args:
            image: Input image (can be color or grayscale)

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize to 0-255 range
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Apply CLAHE
        enhanced = self.feature_extractor.apply_clahe(image, clip_limit=self.clahe_clip)

        # Additional filtering
        filtered = cv2.GaussianBlur(enhanced.astype(np.uint8), (5, 5), 0).astype(np.float32)

        return filtered

    def extract_all_features(self, image):
        """
        Extract all features from preprocessed image.

        Args:
            image: Preprocessed image

        Returns:
            Concatenated feature vector
        """
        features = []

        # LBP features
        lbp_hist, _ = self.feature_extractor.extract_lbp(image, radius=1, n_points=8)
        features.append(lbp_hist)

        # LOOP features
        loop_hist, _ = self.feature_extractor.extract_loop(image, radius=1, n_points=8)
        features.append(loop_hist)

        # LDP features
        ldp_hist, _ = self.feature_extractor.extract_ldp(image, radius=1, n_points=8)
        features.append(ldp_hist)

        # Multi-resolution features
        multi_feat = self.feature_extractor.multi_resolution_analysis(image, scales=[1, 2])
        features.append(multi_feat)

        # Statistical features
        stats = np.array([
            np.mean(image), np.std(image), np.min(image), np.max(image),
            np.median(image), np.percentile(image, 25), np.percentile(image, 75)
        ])
        features.append(stats)

        return np.concatenate(features)

    def train(self, image_paths, labels, verbose=True):
        """
        Train the detector.

        Args:
            image_paths: List of image file paths
            labels: Corresponding labels (0=no hotspot, 1=hotspot)
            verbose: Print progress
        """
        if verbose:
            print("[*] Training Thermal Hotspot Detector")
            print(f"[*] Processing {len(image_paths)} images...")

        # Extract features
        all_features = []
        start_time = time.time()

        for idx, img_path in enumerate(image_paths):
            if verbose and idx % max(1, len(image_paths)//10) == 0:
                print(f"    Progress: {idx}/{len(image_paths)}")

            try:
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"    Warning: Could not load {img_path}")
                    continue

                # Resize to standard size
                image = cv2.resize(image, (256, 256))

                # Preprocess and extract features
                preprocessed = self.preprocess_image(image)
                features = self.extract_all_features(preprocessed)
                all_features.append(features)
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue

        X = np.array(all_features)
        y = np.array(labels[:len(X)])

        extraction_time = time.time() - start_time
        if verbose:
            print(f"[+] Feature extraction completed in {extraction_time:.2f}s")
            print(f"[+] Feature matrix shape: {X.shape}")

        # EGFS Feature Selection
        if verbose:
            print("[*] Applying EGFS Feature Selection...")
        start_time = time.time()

        self.egfs_selector.fit(X, y)
        X_selected = self.egfs_selector.transform(X)

        selection_time = time.time() - start_time
        if verbose:
            print(f"[+] EGFS completed in {selection_time:.2f}s")
            print(f"[+] Selected {X_selected.shape[1]} features from {X.shape[1]}")

        # Standardization
        X_scaled = self.scaler.fit_transform(X_selected)

        # Train SVM Classifier
        if verbose:
            print("[*] Training SVM Classifier...")
        start_time = time.time()

        self.svm_classifier = SVC(kernel=self.svm_kernel, C=1.0, gamma='scale',
                                  probability=True, random_state=42)
        self.svm_classifier.fit(X_scaled, y)

        train_time = time.time() - start_time
        if verbose:
            print(f"[+] SVM training completed in {train_time:.2f}s")

            # Cross-validation
            cv_scores = cross_val_score(self.svm_classifier, X_scaled, y, cv=5)
            print(f"[+] Cross-validation scores: {cv_scores}")
            print(f"[+] Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        self.training_history['extraction_time'] = extraction_time
        self.training_history['selection_time'] = selection_time
        self.training_history['train_time'] = train_time
        self.training_history['features_before'] = X.shape[1]
        self.training_history['features_after'] = X_selected.shape[1]

        return self

    def predict(self, image_path):
        """
        Predict hotspot for single image.

        Args:
            image_path: Path to image

        Returns:
            Prediction (0 or 1) and confidence score
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.resize(image, (256, 256))
        preprocessed = self.preprocess_image(image)
        features = self.extract_all_features(preprocessed)

        X_selected = self.egfs_selector.transform(features.reshape(1, -1))
        X_scaled = self.scaler.transform(X_selected)

        prediction = self.svm_classifier.predict(X_scaled)[0]
        probabilities = self.svm_classifier.predict_proba(X_scaled)[0]
        confidence = np.max(probabilities)

        return prediction, confidence

    def predict_batch(self, image_paths):
        """
        Predict hotspots for multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            Predictions and confidence scores
        """
        predictions = []
        confidences = []

        for img_path in image_paths:
            try:
                pred, conf = self.predict(img_path)
                predictions.append(pred)
                confidences.append(conf)
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")
                predictions.append(-1)
                confidences.append(0.0)

        return np.array(predictions), np.array(confidences)

    def evaluate(self, image_paths, labels):
        """
        Evaluate detector on dataset.

        Args:
            image_paths: List of image paths
            labels: Corresponding labels

        Returns:
            Evaluation metrics
        """
        predictions, confidences = self.predict_batch(image_paths)

        # Filter valid predictions
        valid_mask = predictions != -1
        predictions = predictions[valid_mask]
        labels_valid = np.array(labels)[valid_mask]

        cm = confusion_matrix(labels_valid, predictions)

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }

        return metrics

    def get_feature_importance(self):
        """Get EGFS feature importance scores."""
        return self.egfs_selector.scores, self.egfs_selector.selected_features


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("THERMAL HOTSPOT DETECTION - REAL-TIME ANALYSIS SYSTEM")
    print("=" * 70)
    print()
