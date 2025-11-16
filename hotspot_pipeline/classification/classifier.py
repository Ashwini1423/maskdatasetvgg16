"""
Hotspot Classification Module
==============================
Step 5: Classification (~5ms target)
- SVM/Neural Network with fused features
- Confidence scoring
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pickle


class SVMClassifier:
    """
    Support Vector Machine classifier optimized for real-time hotspot detection.
    Includes confidence scoring through probability calibration.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        probability: bool = True,
        cache_size: int = 500
    ):
        """
        Initialize SVM classifier.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Enable probability estimates
            cache_size: Kernel cache size (MB)
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.cache_size = cache_size

        # Initialize SVM
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            cache_size=cache_size
        )

        self.is_fitted = False
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Train SVM classifier.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            Self
        """
        self.svm.fit(X, y)
        self.classes_ = self.svm.classes_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        return self.svm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        if not self.probability:
            raise ValueError("Probability estimation not enabled")

        return self.svm.predict_proba(X)

    def predict_with_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            (predictions, confidence_scores)
        """
        predictions = self.predict(X)

        if self.probability:
            probabilities = self.predict_proba(X)
            confidence = np.max(probabilities, axis=1)
        else:
            # Use decision function as confidence
            decision = self.svm.decision_function(X)
            if len(self.classes_) == 2:
                confidence = np.abs(decision)
            else:
                confidence = np.max(decision, axis=1)

        return predictions, confidence

    def save(self, filepath: str):
        """Save classifier to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'SVMClassifier':
        """Load classifier from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class NeuralNetworkClassifier:
    """
    Lightweight neural network classifier for real-time inference.
    Optimized for speed with minimal layers.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu',
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50
    ):
        """
        Initialize neural network classifier.

        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = None
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = None

    def _build_model(self, input_dim: int, n_classes: int):
        """
        Build neural network model.

        Args:
            input_dim: Input feature dimension
            n_classes: Number of classes
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow required for NeuralNetworkClassifier")

        # Build sequential model
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            model.add(layers.Dense(hidden_dim, activation=self.activation))
            model.add(layers.Dropout(self.dropout_rate))

        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )

        self.model = model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> 'NeuralNetworkClassifier':
        """
        Train neural network classifier.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            validation_split: Validation data fraction
            verbose: Verbosity level

        Returns:
            Self
        """
        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Build model
        self._build_model(X.shape[1], self.n_classes_)

        # Train model
        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            verbose=verbose
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        probabilities = self.model.predict(X, verbose=0)

        if self.n_classes_ == 2:
            predictions = (probabilities > 0.5).astype(int).flatten()
        else:
            predictions = np.argmax(probabilities, axis=1)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        probabilities = self.model.predict(X, verbose=0)

        if self.n_classes_ == 2:
            # Binary classification: return both class probabilities
            prob_class_1 = probabilities.flatten()
            prob_class_0 = 1 - prob_class_1
            probabilities = np.column_stack([prob_class_0, prob_class_1])

        return probabilities

    def predict_with_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            (predictions, confidence_scores)
        """
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)
        confidence = np.max(probabilities, axis=1)

        return predictions, confidence

    def save(self, filepath: str):
        """Save model to file."""
        if self.model:
            self.model.save(filepath)

    def load(self, filepath: str):
        """Load model from file."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow required for NeuralNetworkClassifier")

        self.model = keras.models.load_model(filepath)
        self.is_fitted = True


class HotspotClassifier:
    """
    Unified hotspot classifier supporting both SVM and Neural Network.
    Provides consistent interface with confidence scoring.
    """

    def __init__(
        self,
        classifier_type: str = 'svm',
        **kwargs
    ):
        """
        Initialize hotspot classifier.

        Args:
            classifier_type: Type of classifier ('svm' or 'neural_network')
            **kwargs: Classifier-specific parameters
        """
        self.classifier_type = classifier_type

        if classifier_type == 'svm':
            self.classifier = SVMClassifier(**kwargs)
        elif classifier_type == 'neural_network':
            self.classifier = NeuralNetworkClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'HotspotClassifier':
        """
        Train classifier.

        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters

        Returns:
            Self
        """
        self.classifier.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.classifier.predict_proba(X)

    def predict_with_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Args:
            X: Input features

        Returns:
            (predictions, confidence_scores)
        """
        return self.classifier.predict_with_confidence(X)

    def classify_hotspot(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Dict[str, any]:
        """
        Classify hotspot with detailed results.

        Args:
            features: Input feature vector
            confidence_threshold: Minimum confidence for positive detection

        Returns:
            Classification results dictionary
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction, confidence = self.predict_with_confidence(features)
        probabilities = self.predict_proba(features)

        result = {
            'is_hotspot': bool(prediction[0]),
            'class': int(prediction[0]),
            'confidence': float(confidence[0]),
            'probabilities': probabilities[0].tolist(),
            'meets_threshold': confidence[0] >= confidence_threshold
        }

        return result

    def save(self, filepath: str):
        """Save classifier to file."""
        self.classifier.save(filepath)

    def load(self, filepath: str):
        """Load classifier from file."""
        if self.classifier_type == 'svm':
            self.classifier = SVMClassifier.load(filepath)
        else:
            self.classifier.load(filepath)

    def __call__(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Allow classifier to be called as a function."""
        return self.predict_with_confidence(X)
