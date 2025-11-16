"""
Fast Feature Extraction Module
================================
Step 2: Fast Feature Extraction (~8ms target)
- Parallel computation of LBP, LDP, LOOP
- Use lookup tables for common patterns
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import time


class LBPExtractor:
    """Local Binary Patterns (LBP) descriptor with optimization."""

    def __init__(self, radius: int = 1, n_points: int = 8, method: str = 'uniform'):
        """
        Initialize LBP extractor.

        Args:
            radius: Radius of circular sampling
            n_points: Number of sampling points
            method: LBP variant ('uniform', 'default', 'var')
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method

        # Precompute lookup table for uniform patterns
        if method == 'uniform':
            self.lut = self._create_uniform_lut()

    def _create_uniform_lut(self) -> np.ndarray:
        """Create lookup table for uniform LBP patterns."""
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Count bit transitions
            transitions = bin(i ^ (i >> 1 | (i & 1) << 7)).count('1')
            if transitions <= 2:
                lut[i] = bin(i).count('1')
            else:
                lut[i] = self.n_points + 1
        return lut

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from image.

        Args:
            image: Input grayscale image

        Returns:
            LBP histogram features
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = image.shape
        lbp_image = np.zeros((height, width), dtype=np.uint8)

        # Optimized LBP computation
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                center = image[i, j]
                pattern = 0

                # Sample circular neighbors
                for p in range(self.n_points):
                    angle = 2 * np.pi * p / self.n_points
                    x = j + self.radius * np.cos(angle)
                    y = i - self.radius * np.sin(angle)

                    # Bilinear interpolation
                    x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
                    x_ceil, y_ceil = int(np.ceil(x)), int(np.ceil(y))

                    if 0 <= x_floor < width and 0 <= y_floor < height:
                        # Simplified bilinear interpolation
                        neighbor = image[y_floor, x_floor]
                        if neighbor >= center:
                            pattern |= (1 << p)

                lbp_image[i, j] = pattern

        # Apply uniform LBP mapping if needed
        if self.method == 'uniform':
            lbp_image = self.lut[lbp_image]

        # Compute histogram
        hist_range = self.n_points + 2 if self.method == 'uniform' else 256
        histogram, _ = np.histogram(lbp_image, bins=hist_range, range=(0, hist_range))

        # Normalize
        histogram = histogram.astype(np.float32)
        histogram /= (histogram.sum() + 1e-7)

        return histogram


class LDPExtractor:
    """Local Directional Patterns (LDP) descriptor."""

    def __init__(self, k: int = 3, mask_size: int = 3):
        """
        Initialize LDP extractor.

        Args:
            k: Number of top directional responses to encode
            mask_size: Kirsch mask size
        """
        self.k = k
        self.mask_size = mask_size

        # Kirsch compass masks (8 directions)
        self.kirsch_masks = self._create_kirsch_masks()

    def _create_kirsch_masks(self) -> List[np.ndarray]:
        """Create 8 Kirsch compass masks for edge detection."""
        masks = []

        # Base mask (0 degrees - East)
        base = np.array([
            [-3, -3, 5],
            [-3, 0, 5],
            [-3, -3, 5]
        ], dtype=np.float32)

        # Rotate to get all 8 directions
        for i in range(8):
            angle = i * 45
            center = (1, 1)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(base, rotation_matrix, (3, 3))
            masks.append(rotated)

        return masks

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LDP features from image.

        Args:
            image: Input grayscale image

        Returns:
            LDP histogram features
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32)
        height, width = image.shape

        # Apply Kirsch masks to get directional responses
        responses = np.zeros((8, height, width), dtype=np.float32)
        for i, mask in enumerate(self.kirsch_masks):
            responses[i] = cv2.filter2D(image, -1, mask)

        # Encode LDP patterns
        ldp_image = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Get top-k directional responses
                response_values = responses[:, i, j]
                top_k_indices = np.argsort(response_values)[-self.k:]

                # Encode as binary pattern
                pattern = 0
                for idx in top_k_indices:
                    pattern |= (1 << idx)

                ldp_image[i, j] = pattern

        # Compute histogram
        histogram, _ = np.histogram(ldp_image, bins=256, range=(0, 256))

        # Normalize
        histogram = histogram.astype(np.float32)
        histogram /= (histogram.sum() + 1e-7)

        return histogram


class LOOPExtractor:
    """Local Optimal Oriented Pattern (LOOP) descriptor."""

    def __init__(self, radius: int = 1, n_points: int = 8):
        """
        Initialize LOOP extractor.

        Args:
            radius: Radius of circular sampling
            n_points: Number of sampling points
        """
        self.radius = radius
        self.n_points = n_points

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LOOP features from image.

        Args:
            image: Input grayscale image

        Returns:
            LOOP histogram features
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

        # Compute magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)

        height, width = image.shape
        loop_image = np.zeros((height, width), dtype=np.uint8)

        # Compute LOOP patterns
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                center_orientation = orientation[i, j]
                pattern = 0

                # Sample circular neighbors
                for p in range(self.n_points):
                    angle = 2 * np.pi * p / self.n_points
                    x = j + self.radius * np.cos(angle)
                    y = i - self.radius * np.sin(angle)

                    # Get neighbor orientation
                    x_int, y_int = int(round(x)), int(round(y))

                    if 0 <= x_int < width and 0 <= y_int < height:
                        neighbor_orientation = orientation[y_int, x_int]

                        # Compare orientations
                        diff = abs(neighbor_orientation - center_orientation)
                        if diff > np.pi:
                            diff = 2 * np.pi - diff

                        # Encode based on optimal orientation
                        if diff < np.pi / 4:  # Similar orientation
                            pattern |= (1 << p)

                loop_image[i, j] = pattern

        # Compute histogram
        histogram, _ = np.histogram(loop_image, bins=256, range=(0, 256))

        # Normalize
        histogram = histogram.astype(np.float32)
        histogram /= (histogram.sum() + 1e-7)

        return histogram


class FastFeatureExtractor:
    """
    Fast parallel feature extraction combining LBP, LDP, and LOOP.
    Target: ~8ms processing time.
    """

    def __init__(
        self,
        use_lbp: bool = True,
        use_ldp: bool = True,
        use_loop: bool = True,
        parallel: bool = True
    ):
        """
        Initialize fast feature extractor.

        Args:
            use_lbp: Enable LBP extraction
            use_ldp: Enable LDP extraction
            use_loop: Enable LOOP extraction
            parallel: Enable parallel computation
        """
        self.use_lbp = use_lbp
        self.use_ldp = use_ldp
        self.use_loop = use_loop
        self.parallel = parallel

        # Initialize extractors
        self.lbp_extractor = LBPExtractor() if use_lbp else None
        self.ldp_extractor = LDPExtractor() if use_ldp else None
        self.loop_extractor = LOOPExtractor() if use_loop else None

        # Thread pool for parallel extraction
        if parallel:
            self.executor = ThreadPoolExecutor(max_workers=3)

    def extract_parallel(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features in parallel.

        Args:
            image: Input image

        Returns:
            Dictionary of feature histograms
        """
        features = {}
        futures = {}

        if self.use_lbp and self.lbp_extractor:
            futures['lbp'] = self.executor.submit(self.lbp_extractor.extract, image)

        if self.use_ldp and self.ldp_extractor:
            futures['ldp'] = self.executor.submit(self.ldp_extractor.extract, image)

        if self.use_loop and self.loop_extractor:
            futures['loop'] = self.executor.submit(self.loop_extractor.extract, image)

        # Collect results
        for name, future in futures.items():
            features[name] = future.result()

        return features

    def extract_sequential(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features sequentially.

        Args:
            image: Input image

        Returns:
            Dictionary of feature histograms
        """
        features = {}

        if self.use_lbp and self.lbp_extractor:
            features['lbp'] = self.lbp_extractor.extract(image)

        if self.use_ldp and self.ldp_extractor:
            features['ldp'] = self.ldp_extractor.extract(image)

        if self.use_loop and self.loop_extractor:
            features['loop'] = self.loop_extractor.extract(image)

        return features

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract and concatenate all features.

        Args:
            image: Input image

        Returns:
            Concatenated feature vector
        """
        if self.parallel:
            features_dict = self.extract_parallel(image)
        else:
            features_dict = self.extract_sequential(image)

        # Concatenate all features
        feature_list = []
        for name in ['lbp', 'ldp', 'loop']:
            if name in features_dict:
                feature_list.append(features_dict[name])

        if not feature_list:
            raise ValueError("No features extracted. Enable at least one descriptor.")

        concatenated = np.concatenate(feature_list)
        return concatenated

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow extractor to be called as a function."""
        return self.extract(image)

    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, 'executor') and self.parallel:
            self.executor.shutdown(wait=True)
