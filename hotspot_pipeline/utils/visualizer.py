"""Visualization utilities for results."""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class ResultVisualizer:
    """Visualize hotspot detection results."""

    def __init__(
        self,
        window_name: str = "Hotspot Detection",
        font_scale: float = 0.6,
        thickness: int = 2
    ):
        """
        Initialize result visualizer.

        Args:
            window_name: Name of the display window
            font_scale: Font scale for text
            thickness: Line thickness
        """
        self.window_name = window_name
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_hotspot_overlay(
        self,
        image: np.ndarray,
        result: Dict,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw hotspot detection overlay on image.

        Args:
            image: Input image
            result: Detection result dictionary
            show_confidence: Show confidence score

        Returns:
            Image with overlay
        """
        overlay = image.copy()

        # Choose color based on detection
        if result['is_hotspot'] and result['meets_threshold']:
            color = (0, 0, 255)  # Red for hotspot
            label = "HOTSPOT"
        else:
            color = (0, 255, 0)  # Green for normal
            label = "NORMAL"

        # Draw label
        text = f"{label}"
        if show_confidence:
            text += f" ({result['confidence']:.2%})"

        # Add background rectangle for text
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
        text_x, text_y = 10, 30

        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.thickness
        )

        return overlay

    def draw_performance_info(
        self,
        image: np.ndarray,
        processing_time: float,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw performance information on image.

        Args:
            image: Input image
            processing_time: Processing time in ms
            fps: Frames per second

        Returns:
            Image with performance info
        """
        overlay = image.copy()
        h, w = image.shape[:2]

        # Performance text
        perf_text = f"Time: {processing_time:.1f}ms"
        if fps is not None:
            perf_text += f" | FPS: {fps:.1f}"

        # Draw text in bottom-left corner
        text_size = cv2.getTextSize(perf_text, self.font, self.font_scale, self.thickness)[0]
        text_x, text_y = 10, h - 10

        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            overlay,
            perf_text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.thickness
        )

        return overlay

    def create_result_display(
        self,
        image: np.ndarray,
        result: Dict,
        processing_time: float,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Create complete result display.

        Args:
            image: Input image
            result: Detection result
            processing_time: Processing time in ms
            fps: Frames per second

        Returns:
            Complete visualization
        """
        # Draw hotspot overlay
        display = self.draw_hotspot_overlay(image, result, show_confidence=True)

        # Draw performance info
        display = self.draw_performance_info(display, processing_time, fps)

        return display

    def show(self, image: np.ndarray, wait_key: int = 1) -> int:
        """
        Display image in window.

        Args:
            image: Image to display
            wait_key: Wait time in ms

        Returns:
            Key code pressed
        """
        cv2.imshow(self.window_name, image)
        return cv2.waitKey(wait_key)

    def close(self):
        """Close all windows."""
        cv2.destroyAllWindows()
