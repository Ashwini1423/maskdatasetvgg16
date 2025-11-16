"""
Real-Time Hotspot Classification Pipeline
==========================================
Optimized processing chain combining all components.
Target: ~30ms per frame with 94-97% accuracy
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import time

from .preprocessing.preprocessor import ThermalPreprocessor
from .feature_extraction.extractors import FastFeatureExtractor
from .feature_selection.egfs import EGFSSelector, AdaptiveFeatureSelector
from .multi_scale.mrtp import MRTPAnalyzer
from .classification.classifier import HotspotClassifier
from .utils.timer import PerformanceTimer
from .utils.visualizer import ResultVisualizer


class RealTimeHotspotPipeline:
    """
    Complete real-time hotspot classification pipeline.

    Processing stages:
    1. Preprocessing (~2ms) - Adaptive histogram equalization, noise reduction
    2. Feature Extraction (~8ms) - Parallel LBP, LDP, LOOP
    3. Feature Selection (~3ms) - EGFS with uniform region detection
    4. Multi-Scale Analysis (~12ms) - MRTP pyramid with early termination
    5. Classification (~5ms) - SVM/NN with confidence scoring

    Total: ~30ms per frame
    """

    def __init__(
        self,
        classifier_type: str = 'svm',
        use_feature_selection: bool = True,
        use_multi_scale: bool = True,
        enable_performance_tracking: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize real-time hotspot pipeline.

        Args:
            classifier_type: Type of classifier ('svm' or 'neural_network')
            use_feature_selection: Enable EGFS feature selection
            use_multi_scale: Enable MRTP multi-scale analysis
            enable_performance_tracking: Enable performance monitoring
            confidence_threshold: Confidence threshold for detection
        """
        self.classifier_type = classifier_type
        self.use_feature_selection = use_feature_selection
        self.use_multi_scale = use_multi_scale
        self.enable_performance_tracking = enable_performance_tracking
        self.confidence_threshold = confidence_threshold

        # Initialize pipeline components
        self.preprocessor = ThermalPreprocessor(
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            denoise_h=10.0
        )

        self.feature_extractor = FastFeatureExtractor(
            use_lbp=True,
            use_ldp=True,
            use_loop=True,
            parallel=True
        )

        self.feature_selector = None  # To be initialized after training
        self.adaptive_selector = None

        self.multi_scale_analyzer = MRTPAnalyzer(
            scales=[1.0, 0.5, 0.25],
            n_levels=3,
            early_termination=True
        ) if use_multi_scale else None

        self.classifier = HotspotClassifier(
            classifier_type=classifier_type
        )

        # Performance tracking
        self.timer = PerformanceTimer() if enable_performance_tracking else None
        self.visualizer = ResultVisualizer()

        self.is_trained = False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1: Preprocessing (~2ms target).

        Args:
            image: Input thermal image

        Returns:
            Preprocessed image
        """
        if self.timer:
            with self.timer.measure('1_preprocessing'):
                preprocessed = self.preprocessor.preprocess(image, denoise_method='bilateral')
        else:
            preprocessed = self.preprocessor.preprocess(image, denoise_method='bilateral')

        return preprocessed

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Step 2: Fast Feature Extraction (~8ms target).

        Args:
            image: Preprocessed image

        Returns:
            Feature vector
        """
        if self.timer:
            with self.timer.measure('2_feature_extraction'):
                features = self.feature_extractor.extract(image)
        else:
            features = self.feature_extractor.extract(image)

        return features

    def select_features(
        self,
        features: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Step 3: Intelligent Feature Selection (~3ms target).

        Args:
            features: Input feature vector
            image: Optional original image for uniform detection

        Returns:
            Selected features
        """
        if not self.use_feature_selection or self.adaptive_selector is None:
            return features

        if self.timer:
            with self.timer.measure('3_feature_selection'):
                selected = self.adaptive_selector.select_features(features, image)
        else:
            selected = self.adaptive_selector.select_features(features, image)

        return selected

    def analyze_multiscale(self, image: np.ndarray) -> np.ndarray:
        """
        Step 4: Multi-Scale Analysis (~12ms target).

        Args:
            image: Preprocessed image

        Returns:
            Multi-scale features
        """
        if not self.use_multi_scale or self.multi_scale_analyzer is None:
            return np.array([])

        if self.timer:
            with self.timer.measure('4_multiscale_analysis'):
                mrtp_features = self.multi_scale_analyzer.extract_mrtp_features(image)
        else:
            mrtp_features = self.multi_scale_analyzer.extract_mrtp_features(image)

        return mrtp_features

    def classify(self, features: np.ndarray) -> Dict:
        """
        Step 5: Classification (~5ms target).

        Args:
            features: Combined feature vector

        Returns:
            Classification result
        """
        if self.timer:
            with self.timer.measure('5_classification'):
                result = self.classifier.classify_hotspot(
                    features,
                    confidence_threshold=self.confidence_threshold
                )
        else:
            result = self.classifier.classify_hotspot(
                features,
                confidence_threshold=self.confidence_threshold
            )

        return result

    def process_frame(self, image: np.ndarray) -> Dict:
        """
        Process single frame through complete pipeline.

        Args:
            image: Input thermal image

        Returns:
            Detection result dictionary with:
                - is_hotspot: Boolean detection result
                - class: Class label
                - confidence: Confidence score
                - probabilities: Class probabilities
                - meets_threshold: Whether confidence meets threshold
                - processing_time: Time taken (if tracking enabled)
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before processing frames")

        start_time = time.perf_counter()

        # Step 1: Preprocessing
        preprocessed = self.preprocess(image)

        # Step 2: Feature Extraction
        features = self.extract_features(preprocessed)

        # Step 3: Feature Selection
        if self.use_feature_selection:
            selected_features = self.select_features(features, preprocessed)
        else:
            selected_features = features

        # Step 4: Multi-Scale Analysis (optional)
        if self.use_multi_scale:
            mrtp_features = self.analyze_multiscale(preprocessed)

            # Combine features if multi-scale produced results
            if len(mrtp_features) > 0:
                combined_features = np.concatenate([selected_features, mrtp_features])
            else:
                combined_features = selected_features
        else:
            combined_features = selected_features

        # Step 5: Classification
        result = self.classify(combined_features)

        # Add processing time
        processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        result['processing_time'] = processing_time

        return result

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the pipeline on training data.

        Args:
            X_train: Training images (n_samples, height, width) or feature arrays
            y_train: Training labels (n_samples,)
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Training metrics
        """
        print("Training pipeline...")

        # Extract features from training images if needed
        if X_train.ndim > 2:
            print("Extracting features from training images...")
            train_features = []
            for i, img in enumerate(X_train):
                if i % 100 == 0:
                    print(f"Processing image {i}/{len(X_train)}...")

                # Preprocess and extract features
                preprocessed = self.preprocess(img)
                features = self.extract_features(preprocessed)
                train_features.append(features)

            X_train_features = np.array(train_features)
        else:
            X_train_features = X_train

        # Train feature selector if enabled
        if self.use_feature_selection:
            print("Training feature selector...")
            self.feature_selector = EGFSSelector(n_features=None)
            X_train_selected = self.feature_selector.fit_transform(X_train_features, y_train)
            self.adaptive_selector = AdaptiveFeatureSelector(self.feature_selector)
        else:
            X_train_selected = X_train_features

        # Train classifier
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train_selected, y_train, **kwargs)

        self.is_trained = True

        # Compute training accuracy
        train_pred = self.classifier.predict(X_train_selected)
        train_accuracy = np.mean(train_pred == y_train)

        metrics = {'train_accuracy': train_accuracy}

        # Validation accuracy if provided
        if X_val is not None and y_val is not None:
            val_features = []
            for img in X_val:
                preprocessed = self.preprocess(img)
                features = self.extract_features(preprocessed)
                if self.use_feature_selection:
                    features = self.feature_selector.transform(features.reshape(1, -1)).flatten()
                val_features.append(features)

            X_val_features = np.array(val_features)
            val_pred = self.classifier.predict(X_val_features)
            val_accuracy = np.mean(val_pred == y_val)
            metrics['val_accuracy'] = val_accuracy

        print(f"\nTraining complete!")
        print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
        if 'val_accuracy' in metrics:
            print(f"Val Accuracy: {metrics['val_accuracy']:.2%}")

        return metrics

    def get_performance_report(self) -> Dict:
        """
        Get performance report.

        Returns:
            Performance statistics
        """
        if not self.timer:
            raise ValueError("Performance tracking not enabled")

        return self.timer.get_statistics()

    def print_performance_report(self):
        """Print detailed performance report."""
        if not self.timer:
            raise ValueError("Performance tracking not enabled")

        self.timer.print_report()

    def save_model(self, filepath: str):
        """
        Save trained model.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before saving")

        self.classifier.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load trained model.

        Args:
            filepath: Path to model file
        """
        self.classifier.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def process_video(
        self,
        video_source: int = 0,
        display: bool = True,
        save_output: Optional[str] = None
    ):
        """
        Process video stream in real-time.

        Args:
            video_source: Video source (0 for webcam, or file path)
            display: Display results
            save_output: Path to save output video (optional)
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before processing video")

        cap = cv2.VideoCapture(video_source)

        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        else:
            out = None

        frame_count = 0
        fps_tracker = []

        print("Processing video... Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = self.process_frame(frame)

                # Track FPS
                fps_tracker.append(result['processing_time'])
                if len(fps_tracker) > 30:
                    fps_tracker.pop(0)
                avg_fps = 1000 / (sum(fps_tracker) / len(fps_tracker))

                # Visualize results
                if display or save_output:
                    display_frame = self.visualizer.create_result_display(
                        frame,
                        result,
                        result['processing_time'],
                        avg_fps
                    )

                    if display:
                        key = self.visualizer.show(display_frame, wait_key=1)
                        if key & 0xFF == ord('q'):
                            break

                    if save_output and out:
                        out.write(display_frame)

                frame_count += 1

        finally:
            cap.release()
            if out:
                out.release()
            if display:
                self.visualizer.close()

            print(f"\nProcessed {frame_count} frames")
            if self.timer:
                self.print_performance_report()
