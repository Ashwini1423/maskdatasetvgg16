"""
Multi-Resolution Texture Patterns (MRTP)
=========================================
Step 4: Multi-Scale Analysis (~12ms target)
- MRTP pyramid processing
- Early termination for obvious non-hotspots
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class MRTPAnalyzer:
    """
    Multi-Resolution Texture Pattern analyzer with pyramid processing.
    Includes early termination for non-hotspot regions.
    """

    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        n_levels: int = 3,
        early_termination: bool = True,
        intensity_threshold: float = 0.3,
        texture_threshold: float = 0.1
    ):
        """
        Initialize MRTP analyzer.

        Args:
            scales: Scale factors for pyramid levels
            n_levels: Number of pyramid levels
            early_termination: Enable early termination for non-hotspots
            intensity_threshold: Minimum intensity for hotspot candidate
            texture_threshold: Minimum texture variance for processing
        """
        self.scales = scales[:n_levels]
        self.n_levels = n_levels
        self.early_termination = early_termination
        self.intensity_threshold = intensity_threshold
        self.texture_threshold = texture_threshold

    def build_gaussian_pyramid(
        self,
        image: np.ndarray,
        levels: int
    ) -> List[np.ndarray]:
        """
        Build Gaussian pyramid for multi-scale analysis.

        Args:
            image: Input image
            levels: Number of pyramid levels

        Returns:
            List of images at different scales
        """
        pyramid = [image]

        for i in range(levels - 1):
            # Downsample image
            downsampled = cv2.pyrDown(pyramid[-1])
            pyramid.append(downsampled)

        return pyramid

    def build_laplacian_pyramid(
        self,
        image: np.ndarray,
        levels: int
    ) -> List[np.ndarray]:
        """
        Build Laplacian pyramid for edge information at multiple scales.

        Args:
            image: Input image
            levels: Number of pyramid levels

        Returns:
            List of Laplacian images
        """
        gaussian_pyramid = self.build_gaussian_pyramid(image, levels + 1)
        laplacian_pyramid = []

        for i in range(levels):
            # Expand lower level
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1])

            # Resize if necessary (handle size mismatch)
            if expanded.shape != gaussian_pyramid[i].shape:
                expanded = cv2.resize(
                    expanded,
                    (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
                )

            # Compute Laplacian
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)

        return laplacian_pyramid

    def check_early_termination(self, image: np.ndarray) -> bool:
        """
        Check if region should be terminated early (obvious non-hotspot).

        Args:
            image: Input image region

        Returns:
            True if should terminate (non-hotspot), False if should continue
        """
        if not self.early_termination:
            return False

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check intensity (hotspots should be bright in thermal images)
        normalized = image.astype(np.float32) / 255.0
        mean_intensity = np.mean(normalized)

        if mean_intensity < self.intensity_threshold:
            return True  # Too dark, likely not a hotspot

        # Check texture variance (uniform regions are less interesting)
        variance = np.var(normalized)

        if variance < self.texture_threshold:
            return True  # Too uniform, likely background

        return False  # Continue processing

    def extract_texture_features(
        self,
        image: np.ndarray,
        use_glcm: bool = False
    ) -> np.ndarray:
        """
        Extract texture features from image.

        Args:
            image: Input image
            use_glcm: Use GLCM features (slower but more accurate)

        Returns:
            Texture feature vector
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = []

        # Statistical features (fast)
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.max(image) - np.min(image))  # Range

        # Gradient features
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))

        # Edge density
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        if use_glcm:
            # GLCM features (optional, more computationally expensive)
            glcm_features = self._compute_glcm_features(image)
            features.extend(glcm_features)

        return np.array(features, dtype=np.float32)

    def _compute_glcm_features(self, image: np.ndarray) -> List[float]:
        """
        Compute Gray-Level Co-occurrence Matrix features.

        Args:
            image: Input grayscale image

        Returns:
            List of GLCM features
        """
        # Simplified GLCM computation (for speed)
        # In production, use skimage.feature.greycomatrix for full GLCM

        # Quantize to reduce computation
        quantized = (image // 32).astype(np.uint8)  # 8 levels

        # Compute simple co-occurrence statistics
        # Horizontal pairs
        pairs_h = quantized[:, :-1] * 8 + quantized[:, 1:]
        hist_h, _ = np.histogram(pairs_h, bins=64, range=(0, 64))

        # Vertical pairs
        pairs_v = quantized[:-1, :] * 8 + quantized[1:, :]
        hist_v, _ = np.histogram(pairs_v, bins=64, range=(0, 64))

        # Combine histograms
        combined = (hist_h + hist_v) / 2.0
        combined = combined / (combined.sum() + 1e-7)

        # Compute GLCM-like features
        contrast = np.sum(combined * np.arange(64)**2)
        energy = np.sum(combined**2)
        entropy = -np.sum(combined * np.log(combined + 1e-7))

        return [contrast, energy, entropy]

    def analyze_multiscale(
        self,
        image: np.ndarray,
        pyramid_type: str = 'gaussian'
    ) -> Dict[str, np.ndarray]:
        """
        Perform multi-scale texture analysis.

        Args:
            image: Input image
            pyramid_type: Type of pyramid ('gaussian' or 'laplacian')

        Returns:
            Dictionary containing multi-scale features
        """
        # Early termination check
        if self.check_early_termination(image):
            return {
                'is_hotspot_candidate': False,
                'features': np.zeros(0),
                'scales_processed': 0
            }

        # Build pyramid
        if pyramid_type == 'gaussian':
            pyramid = self.build_gaussian_pyramid(image, self.n_levels)
        elif pyramid_type == 'laplacian':
            pyramid = self.build_laplacian_pyramid(image, self.n_levels)
        else:
            raise ValueError(f"Unknown pyramid type: {pyramid_type}")

        # Extract features at each scale
        multiscale_features = []

        for level, scaled_image in enumerate(pyramid):
            # Extract texture features
            features = self.extract_texture_features(scaled_image)
            multiscale_features.append(features)

            # Early termination at coarser scales
            if self.early_termination and level > 0:
                if self.check_early_termination(scaled_image):
                    # Non-hotspot detected at coarser scale
                    return {
                        'is_hotspot_candidate': False,
                        'features': np.concatenate(multiscale_features),
                        'scales_processed': level + 1
                    }

        # Concatenate all scale features
        all_features = np.concatenate(multiscale_features)

        return {
            'is_hotspot_candidate': True,
            'features': all_features,
            'scales_processed': len(pyramid)
        }

    def extract_mrtp_features(
        self,
        image: np.ndarray,
        combine_pyramids: bool = True
    ) -> np.ndarray:
        """
        Extract Multi-Resolution Texture Pattern features.

        Args:
            image: Input image
            combine_pyramids: Combine Gaussian and Laplacian features

        Returns:
            MRTP feature vector
        """
        # Analyze with Gaussian pyramid
        gaussian_result = self.analyze_multiscale(image, pyramid_type='gaussian')

        if not gaussian_result['is_hotspot_candidate']:
            # Return empty features for non-hotspot
            return np.zeros(0)

        features = [gaussian_result['features']]

        # Optionally add Laplacian pyramid features
        if combine_pyramids:
            laplacian_result = self.analyze_multiscale(image, pyramid_type='laplacian')
            if laplacian_result['is_hotspot_candidate']:
                features.append(laplacian_result['features'])

        # Concatenate all features
        mrtp_features = np.concatenate(features)

        return mrtp_features

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow analyzer to be called as a function."""
        return self.extract_mrtp_features(image)


class PyramidProcessor:
    """Helper class for efficient pyramid processing."""

    @staticmethod
    def compute_pyramid_dimensions(
        image_shape: Tuple[int, int],
        n_levels: int
    ) -> List[Tuple[int, int]]:
        """
        Compute dimensions at each pyramid level.

        Args:
            image_shape: Original image shape (height, width)
            n_levels: Number of pyramid levels

        Returns:
            List of (height, width) tuples for each level
        """
        dimensions = [image_shape]
        h, w = image_shape

        for _ in range(n_levels - 1):
            h = (h + 1) // 2
            w = (w + 1) // 2
            dimensions.append((h, w))

        return dimensions

    @staticmethod
    def resize_to_pyramid_level(
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to target pyramid level size.

        Args:
            image: Input image
            target_size: Target (height, width)

        Returns:
            Resized image
        """
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
