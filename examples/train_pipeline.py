"""
Example: Training the Real-Time Hotspot Classification Pipeline
================================================================

This script demonstrates how to train the pipeline on thermal image data.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hotspot_pipeline import RealTimeHotspotPipeline


def load_thermal_dataset(data_dir):
    """
    Load thermal image dataset.

    Args:
        data_dir: Directory containing thermal images

    Returns:
        X: Images array
        y: Labels array
    """
    # Example: Load images from directory structure
    # data_dir/
    #   hotspot/
    #   normal/

    images = []
    labels = []

    data_path = Path(data_dir)

    # Load hotspot images (label = 1)
    hotspot_dir = data_path / 'hotspot'
    if hotspot_dir.exists():
        for img_path in hotspot_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(1)

    # Load normal images (label = 0)
    normal_dir = data_path / 'normal'
    if normal_dir.exists():
        for img_path in normal_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(0)

    return np.array(images), np.array(labels)


def main():
    """Main training script."""

    print("=" * 60)
    print("Real-Time Hotspot Classification Pipeline - Training")
    print("=" * 60)

    # Configuration
    DATA_DIR = "path/to/thermal/dataset"  # Update this path
    MODEL_SAVE_PATH = "models/hotspot_classifier.pkl"

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = RealTimeHotspotPipeline(
        classifier_type='svm',  # or 'neural_network'
        use_feature_selection=True,
        use_multi_scale=True,
        enable_performance_tracking=True,
        confidence_threshold=0.7
    )

    # Load dataset
    print(f"\nLoading dataset from {DATA_DIR}...")
    try:
        X, y = load_thermal_dataset(DATA_DIR)
        print(f"Loaded {len(X)} images")
        print(f"Classes: {np.unique(y, return_counts=True)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nUsing synthetic data for demonstration...")

        # Generate synthetic data for demonstration
        n_samples = 100
        X = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
             for _ in range(n_samples)]
        y = np.random.randint(0, 2, n_samples)
        X = np.array(X)

    # Split into train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} images")
    print(f"Val set: {len(X_val)} images")

    # Train pipeline
    print("\n" + "=" * 60)
    print("Training pipeline...")
    print("=" * 60)

    metrics = pipeline.train(
        X_train, y_train,
        X_val, y_val,
        verbose=0  # Set to 1 for training progress
    )

    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
    if 'val_accuracy' in metrics:
        print(f"Val Accuracy: {metrics['val_accuracy']:.2%}")

    # Save model
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_model(MODEL_SAVE_PATH)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
