"""
Example: Real-Time Hotspot Detection Inference
===============================================

This script demonstrates real-time hotspot detection on video streams.
Target: ~30ms per frame processing time
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hotspot_pipeline import RealTimeHotspotPipeline


def process_single_image(pipeline, image_path):
    """
    Process a single thermal image.

    Args:
        pipeline: Trained pipeline
        image_path: Path to image file
    """
    print(f"\nProcessing image: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Process frame
    result = pipeline.process_frame(image)

    # Display results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)
    print(f"Is Hotspot:      {result['is_hotspot']}")
    print(f"Confidence:      {result['confidence']:.2%}")
    print(f"Meets Threshold: {result['meets_threshold']}")
    print(f"Processing Time: {result['processing_time']:.2f} ms")
    print(f"Probabilities:   {result['probabilities']}")

    # Visualize
    display = pipeline.visualizer.create_result_display(
        image,
        result,
        result['processing_time']
    )

    cv2.imshow("Hotspot Detection", display)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video_stream(pipeline, video_source, save_output=None):
    """
    Process video stream in real-time.

    Args:
        pipeline: Trained pipeline
        video_source: Video source (0 for webcam, or file path)
        save_output: Path to save output video (optional)
    """
    print(f"\nProcessing video from: {video_source}")
    print("Press 'q' to quit")

    pipeline.process_video(
        video_source=video_source,
        display=True,
        save_output=save_output
    )


def benchmark_performance(pipeline, n_frames=100):
    """
    Benchmark pipeline performance.

    Args:
        pipeline: Trained pipeline
        n_frames: Number of frames to process
    """
    print(f"\nRunning performance benchmark ({n_frames} frames)...")

    # Generate random test frames
    frames = [
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        for _ in range(n_frames)
    ]

    # Process frames
    for i, frame in enumerate(frames):
        result = pipeline.process_frame(frame)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_frames} frames...")

    # Print performance report
    print("\n" + "=" * 60)
    pipeline.print_performance_report()


def main():
    """Main inference script."""

    parser = argparse.ArgumentParser(
        description="Real-Time Hotspot Detection Inference"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='image',
        choices=['image', 'video', 'webcam', 'benchmark'],
        help='Inference mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input image/video path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/hotspot_classifier.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (for video mode)'
    )
    parser.add_argument(
        '--benchmark-frames',
        type=int,
        default=100,
        help='Number of frames for benchmark'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Real-Time Hotspot Classification Pipeline - Inference")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = RealTimeHotspotPipeline(
        classifier_type='svm',
        use_feature_selection=True,
        use_multi_scale=True,
        enable_performance_tracking=True,
        confidence_threshold=0.7
    )

    # Load trained model
    print(f"Loading model from {args.model}...")
    try:
        pipeline.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: Please train the model first using train_pipeline.py")
        return

    # Run inference based on mode
    if args.mode == 'image':
        if args.input is None:
            print("Error: --input required for image mode")
            return
        process_single_image(pipeline, args.input)

    elif args.mode == 'video':
        if args.input is None:
            print("Error: --input required for video mode")
            return
        process_video_stream(pipeline, args.input, args.output)

    elif args.mode == 'webcam':
        process_video_stream(pipeline, 0, args.output)

    elif args.mode == 'benchmark':
        benchmark_performance(pipeline, args.benchmark_frames)

    print("\nInference complete!")


if __name__ == "__main__":
    main()
