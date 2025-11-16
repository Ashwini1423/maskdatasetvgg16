"""
Thermal Image Preprocessing Module
===================================
Step 1: Preprocessing (~2ms target)
- Adaptive histogram equalization
- Thermal noise reduction
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ThermalPreprocessor:
    """Fast thermal image preprocessor with noise reduction and enhancement."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        denoise_h: float = 10.0,
        denoise_template_size: int = 7,
        denoise_search_size: int = 21
    ):
        """
        Initialize thermal preprocessor.

        Args:
            clahe_clip_limit: Contrast limit for CLAHE
            clahe_tile_size: Grid size for CLAHE
            denoise_h: Filter strength for denoising
            denoise_template_size: Template patch size
            denoise_search_size: Search window size
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_h = denoise_h
        self.denoise_template_size = denoise_template_size
        self.denoise_search_size = denoise_search_size

        # Create CLAHE object (reusable for efficiency)
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )

    def apply_adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input grayscale or thermal image

        Returns:
            Enhanced image with improved contrast
        """
        if len(image.shape) == 3:
            # Convert to grayscale if color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply CLAHE
        enhanced = self.clahe.apply(image)
        return enhanced

    def reduce_thermal_noise(
        self,
        image: np.ndarray,
        method: str = 'fastNlMeans'
    ) -> np.ndarray:
        """
        Apply thermal noise reduction.

        Args:
            image: Input thermal image
            method: Denoising method ('fastNlMeans', 'bilateral', 'gaussian')

        Returns:
            Denoised image
        """
        if method == 'fastNlMeans':
            # Fast Non-Local Means Denoising (best quality)
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    h=self.denoise_h,
                    hColor=self.denoise_h,
                    templateWindowSize=self.denoise_template_size,
                    searchWindowSize=self.denoise_search_size
                )
            else:
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    h=self.denoise_h,
                    templateWindowSize=self.denoise_template_size,
                    searchWindowSize=self.denoise_search_size
                )

        elif method == 'bilateral':
            # Bilateral filter (preserves edges, faster)
            denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        elif method == 'gaussian':
            # Gaussian blur (fastest, but less edge-preserving)
            denoised = cv2.GaussianBlur(image, (5, 5), 0)

        else:
            raise ValueError(f"Unknown denoising method: {method}")

        return denoised

    def preprocess(
        self,
        image: np.ndarray,
        denoise_method: str = 'bilateral'  # Use faster method for real-time
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline (~2ms target).

        Args:
            image: Input thermal image
            denoise_method: Denoising method to use

        Returns:
            Preprocessed thermal image
        """
        # Step 1: Thermal noise reduction (using bilateral for speed)
        denoised = self.reduce_thermal_noise(image, method=denoise_method)

        # Step 2: Adaptive histogram equalization
        enhanced = self.apply_adaptive_histogram_equalization(denoised)

        return enhanced

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow preprocessor to be called as a function."""
        return self.preprocess(image)
