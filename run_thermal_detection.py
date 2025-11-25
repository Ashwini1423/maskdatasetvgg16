"""
Main Execution Script - Real-Time Thermal Hotspot Detection
Run this script in VS Code for complete analysis and visualization.

Usage:
    python run_thermal_detection.py
"""

import numpy as np
import cv2
import os
from pathlib import Path
import sys
from datetime import datetime
import json

from thermal_hotspot_detection import ThermalHotspotDetector, FeatureExtractor
from visualization import ThermalVisualization


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_thermal_dataset():
    """
    Prepare thermal dataset from existing mask dataset.
    Converts mask detection classes to hotspot detection classes.

    Returns:
        training_images, training_labels, test_images, test_labels
    """
    base_path = Path('/home/user/maskdatasetvgg16')
    data_path = base_path / 'data'
    test_path = base_path / 'test'

    print("[*] Preparing Thermal Hotspot Dataset")
    print("[*] Converting existing mask dataset to thermal hotspot dataset...")

    training_images = []
    training_labels = []
    test_images = []
    test_labels = []

    # Map classes: single=hotspot, straight=no-hotspot, multiple=hotspot
    class_mapping = {
        'single': 1,      # Hotspot
        'straight': 0,    # No hotspot
        'multiple': 1     # Hotspot
    }

    # Load training data
    if data_path.exists():
        for class_name, label in class_mapping.items():
            class_path = data_path / class_name
            if class_path.exists():
                for img_file in class_path.glob('*.jpg'):
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            training_images.append(str(img_file))
                            training_labels.append(label)
                    except Exception as e:
                        print(f"    Warning: Could not load {img_file}: {e}")

    # Load test data
    if test_path.exists():
        for class_name, label in class_mapping.items():
            class_path = test_path / class_name
            if class_path.exists():
                for img_file in class_path.glob('*.jpg'):
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            test_images.append(str(img_file))
                            test_labels.append(label)
                    except Exception as e:
                        print(f"    Warning: Could not load {img_file}: {e}")

    print(f"[+] Training samples: {len(training_images)} (Hotspots: {sum(training_labels)}, "
          f"No-hotspots: {len(training_labels) - sum(training_labels)})")
    print(f"[+] Test samples: {len(test_images)} (Hotspots: {sum(test_labels)}, "
          f"No-hotspots: {len(test_labels) - sum(test_labels)})")

    return training_images, training_labels, test_images, test_labels


# ============================================================================
# FEATURE VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_feature_extraction(detector, sample_image_path):
    """
    Visualize feature extraction pipeline on a sample image.

    Args:
        detector: ThermalHotspotDetector instance
        sample_image_path: Path to sample image
    """
    print("\n[*] Visualizing Feature Extraction Pipeline...")

    # Load and preprocess image
    image = cv2.imread(str(sample_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("    Error: Could not load sample image")
        return

    image = cv2.resize(image, (256, 256))
    preprocessed = detector.preprocess_image(image)

    # Extract individual features
    lbp_hist, lbp_map = detector.feature_extractor.extract_lbp(preprocessed)
    loop_hist, loop_map = detector.feature_extractor.extract_loop(preprocessed)
    ldp_hist, ldp_map = detector.feature_extractor.extract_ldp(preprocessed)

    # Visualize
    visualizer = ThermalVisualization()

    # Feature extraction pipeline
    output_path = Path('/home/user/maskdatasetvgg16/plots/01_feature_extraction_pipeline.png')
    output_path.parent.mkdir(exist_ok=True)
    visualizer.plot_feature_extraction_pipeline(
        image, preprocessed, lbp_map, loop_map, ldp_map,
        save_path=output_path
    )

    # Feature histograms
    output_path = Path('/home/user/maskdatasetvgg16/plots/02_feature_histograms.png')
    visualizer.plot_feature_histograms(lbp_hist, loop_hist, ldp_hist,
                                       save_path=output_path)

    print("[+] Feature extraction visualization complete")


def visualize_egfs_results(detector):
    """
    Visualize EGFS feature selection results.

    Args:
        detector: ThermalHotspotDetector instance
    """
    print("\n[*] Visualizing EGFS Feature Selection Results...")

    scores, selected_features = detector.get_feature_importance()
    visualizer = ThermalVisualization()

    output_path = Path('/home/user/maskdatasetvgg16/plots/03_egfs_feature_selection.png')
    output_path.parent.mkdir(exist_ok=True)
    visualizer.plot_egfs_results(scores, selected_features,
                                 n_display=30, save_path=output_path)

    print("[+] EGFS visualization complete")
    print(f"    Total features: {len(scores)}")
    print(f"    Selected features: {len(selected_features)}")
    print(f"    Reduction ratio: {(1 - len(selected_features)/len(scores))*100:.1f}%")


def visualize_training_summary(detector):
    """
    Visualize training summary.

    Args:
        detector: ThermalHotspotDetector instance
    """
    print("\n[*] Visualizing Training Summary...")

    visualizer = ThermalVisualization()
    output_path = Path('/home/user/maskdatasetvgg16/plots/04_training_summary.png')
    output_path.parent.mkdir(exist_ok=True)

    visualizer.plot_training_summary(detector.training_history,
                                    save_path=output_path)

    print("[+] Training summary visualization complete")


def visualize_evaluation_results(metrics):
    """
    Visualize evaluation results.

    Args:
        metrics: Evaluation metrics dictionary
    """
    print("\n[*] Visualizing Evaluation Results...")

    visualizer = ThermalVisualization()

    # Confusion matrix
    output_path = Path('/home/user/maskdatasetvgg16/plots/05_confusion_matrix.png')
    output_path.parent.mkdir(exist_ok=True)
    visualizer.plot_confusion_matrix(metrics['confusion_matrix'],
                                     save_path=output_path)

    # Classification metrics
    output_path = Path('/home/user/maskdatasetvgg16/plots/06_classification_metrics.png')
    visualizer.plot_classification_metrics(metrics, save_path=output_path)

    print("[+] Evaluation visualization complete")


# ============================================================================
# REAL-TIME DETECTION DEMO
# ============================================================================

def real_time_detection_demo(detector, test_images, test_labels):
    """
    Demonstrate real-time detection on sample images.

    Args:
        detector: ThermalHotspotDetector instance
        test_images: List of test image paths
        test_labels: Corresponding labels
    """
    print("\n[*] Running Real-Time Detection Demo...")
    print("    Analyzing sample test images...")

    visualizer = ThermalVisualization()

    # Select sample images
    hotspot_indices = [i for i, label in enumerate(test_labels) if label == 1]
    no_hotspot_indices = [i for i, label in enumerate(test_labels) if label == 0]

    samples_to_show = []
    if hotspot_indices:
        samples_to_show.append(hotspot_indices[0])
    if no_hotspot_indices:
        samples_to_show.append(no_hotspot_indices[0])

    for idx, sample_idx in enumerate(samples_to_show):
        img_path = test_images[sample_idx]
        true_label = test_labels[sample_idx]

        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, (256, 256))

            # Predict
            prediction, confidence = detector.predict(img_path)

            result = "✓ CORRECT" if prediction == true_label else "✗ INCORRECT"
            label_str = "Hotspot" if prediction == 1 else "No Hotspot"

            print(f"    Sample {idx + 1}: {label_str} ({confidence:.2%} confidence) {result}")

            # Visualize
            output_path = Path(f'/home/user/maskdatasetvgg16/plots/07_realtime_detection_{idx}.png')
            output_path.parent.mkdir(exist_ok=True)
            visualizer.plot_real_time_detection(image, prediction, confidence,
                                               save_path=output_path)
        except Exception as e:
            print(f"    Error processing sample: {e}")

    print("[+] Real-time detection demo complete")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""

    print("\n" + "=" * 80)
    print("REAL-TIME THERMAL HOTSPOT DETECTION SYSTEM")
    print("Based on: Real-Time Analysis of Thermal Hotspot Detection Using")
    print("          Multi-Feature Extraction and Enhanced Graph-Based Selection")
    print("=" * 80 + "\n")

    print(f"[*] Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # ====================================================================
        # PHASE 1: DATA PREPARATION
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 1: DATA PREPARATION")
        print("-" * 80)

        train_images, train_labels, test_images, test_labels = prepare_thermal_dataset()

        if len(train_images) == 0:
            print("[!] Error: No training images found")
            return

        # ====================================================================
        # PHASE 2: DETECTOR INITIALIZATION AND TRAINING
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 2: THERMAL HOTSPOT DETECTOR INITIALIZATION & TRAINING")
        print("-" * 80)

        detector = ThermalHotspotDetector(
            clahe_clip=2.0,
            svm_kernel='rbf',
            egfs_features=50
        )

        print("[*] Configuration:")
        print(f"    CLAHE Clip Limit: 2.0")
        print(f"    SVM Kernel: rbf")
        print(f"    EGFS Target Features: 50")

        # Train detector
        detector.train(train_images, train_labels, verbose=True)

        # ====================================================================
        # PHASE 3: FEATURE VISUALIZATION
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 3: FEATURE EXTRACTION VISUALIZATION")
        print("-" * 80)

        if train_images:
            sample_image = train_images[0]
            visualize_feature_extraction(detector, sample_image)

        # ====================================================================
        # PHASE 4: EGFS RESULTS VISUALIZATION
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 4: ENHANCED GRAPH-BASED FEATURE SELECTION RESULTS")
        print("-" * 80)

        visualize_egfs_results(detector)

        # ====================================================================
        # PHASE 5: TRAINING SUMMARY
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 5: TRAINING SUMMARY")
        print("-" * 80)

        visualize_training_summary(detector)

        # ====================================================================
        # PHASE 6: EVALUATION
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 6: MODEL EVALUATION")
        print("-" * 80)

        print("[*] Evaluating on test set...")
        metrics = detector.evaluate(test_images, test_labels)

        print(f"[+] Accuracy: {metrics['accuracy']:.4f}")
        print(f"[+] Precision: {metrics['precision']:.4f}")
        print(f"[+] Recall: {metrics['recall']:.4f}")
        print(f"[+] F1-Score: {metrics['f1_score']:.4f}")

        # ====================================================================
        # PHASE 7: EVALUATION VISUALIZATION
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 7: EVALUATION VISUALIZATION")
        print("-" * 80)

        visualize_evaluation_results(metrics)

        # ====================================================================
        # PHASE 8: REAL-TIME DETECTION DEMO
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 8: REAL-TIME DETECTION DEMONSTRATION")
        print("-" * 80)

        real_time_detection_demo(detector, test_images, test_labels)

        # ====================================================================
        # PHASE 9: SUMMARY REPORT
        # ====================================================================

        print("\n" + "-" * 80)
        print("PHASE 9: SUMMARY REPORT")
        print("-" * 80)

        print("\n[+] SYSTEM PERFORMANCE SUMMARY")
        print(f"    Total Training Samples: {len(train_images)}")
        print(f"    Total Test Samples: {len(test_images)}")
        print(f"    Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")

        feature_reduction = (1 - detector.training_history['features_after'] /
                           detector.training_history['features_before']) * 100
        print(f"\n[+] FEATURE SELECTION SUMMARY")
        print(f"    Original Features: {detector.training_history['features_before']}")
        print(f"    Selected Features: {detector.training_history['features_after']}")
        print(f"    Reduction: {feature_reduction:.1f}%")

        print(f"\n[+] PROCESSING TIME SUMMARY")
        print(f"    Feature Extraction: {detector.training_history['extraction_time']:.2f}s")
        print(f"    EGFS Selection: {detector.training_history['selection_time']:.2f}s")
        print(f"    SVM Training: {detector.training_history['train_time']:.2f}s")
        total_time = (detector.training_history['extraction_time'] +
                     detector.training_history['selection_time'] +
                     detector.training_history['train_time'])
        print(f"    Total Time: {total_time:.2f}s")

        print(f"\n[+] Generated Plots (saved to /plots/):")
        print(f"    01_feature_extraction_pipeline.png")
        print(f"    02_feature_histograms.png")
        print(f"    03_egfs_feature_selection.png")
        print(f"    04_training_summary.png")
        print(f"    05_confusion_matrix.png")
        print(f"    06_classification_metrics.png")
        print(f"    07_realtime_detection_*.png")

        print(f"\n[*] Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n[+] All plots have been generated and are ready for review!")
        print("[+] Check the /plots/ directory for visualization outputs.")

        print("\n" + "=" * 80)
        print("THERMAL HOTSPOT DETECTION SYSTEM - EXECUTION SUCCESSFUL")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[!] Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
