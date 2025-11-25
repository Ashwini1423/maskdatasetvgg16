"""
Unit tests for thermal hotspot detection modules.
Verifies that all components are working correctly without running full pipeline.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

print("[*] Testing Thermal Hotspot Detection Modules")
print("[*] " + "="*70)

# Test 1: Import modules
print("\n[1] Testing Module Imports...")
try:
    from thermal_hotspot_detection import (
        FeatureExtractor, EGFSSelector, ThermalHotspotDetector
    )
    from visualization import ThermalVisualization
    print("[+] All modules imported successfully")
except Exception as e:
    print(f"[!] Import failed: {e}")
    sys.exit(1)

# Test 2: Feature Extractor
print("\n[2] Testing Feature Extractor...")
try:
    extractor = FeatureExtractor()

    # Create synthetic test image
    test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    # Test LBP
    lbp_hist, lbp_map = extractor.extract_lbp(test_image)
    assert lbp_hist.shape[0] == 256, "LBP histogram size incorrect"
    assert lbp_map.shape == test_image.shape, "LBP map shape incorrect"
    print("[+] LBP extraction working")

    # Test LOOP
    loop_hist, loop_map = extractor.extract_loop(test_image)
    assert loop_hist.shape[0] == 256, "LOOP histogram size incorrect"
    assert loop_map.shape == test_image.shape, "LOOP map shape incorrect"
    print("[+] LOOP extraction working")

    # Test LDP
    ldp_hist, ldp_map = extractor.extract_ldp(test_image)
    assert ldp_hist.shape[0] == 256, "LDP histogram size incorrect"
    assert ldp_map.shape == test_image.shape, "LDP map shape incorrect"
    print("[+] LDP extraction working")

    # Test CLAHE
    enhanced = extractor.apply_clahe(test_image)
    assert enhanced.shape == test_image.shape, "CLAHE output shape incorrect"
    print("[+] CLAHE preprocessing working")

    # Test multi-resolution
    multi_feat = extractor.multi_resolution_analysis(test_image)
    assert len(multi_feat) > 0, "Multi-resolution features empty"
    print("[+] Multi-resolution analysis working")

except Exception as e:
    print(f"[!] Feature extraction test failed: {e}")
    sys.exit(1)

# Test 3: EGFS Selector
print("\n[3] Testing EGFS Feature Selector...")
try:
    selector = EGFSSelector(n_features=20)

    # Create synthetic training data
    X_train = np.random.randn(50, 100)  # 50 samples, 100 features
    y_train = np.random.randint(0, 2, 50)  # Binary labels

    # Test fit
    selector.fit(X_train, y_train)
    assert selector.selected_features is not None, "Selected features is None"
    assert len(selector.selected_features) == 20, "Wrong number of selected features"
    print("[+] EGFS fit working")

    # Test transform
    X_transformed = selector.transform(X_train)
    assert X_transformed.shape[1] == 20, "Transformed feature count incorrect"
    print("[+] EGFS transform working")

    # Test fit_transform
    X_ft = selector.fit_transform(X_train, y_train)
    assert X_ft.shape[1] == 20, "Fit-transform feature count incorrect"
    print("[+] EGFS fit_transform working")

except Exception as e:
    print(f"[!] EGFS test failed: {e}")
    sys.exit(1)

# Test 4: ThermalHotspotDetector
print("\n[4] Testing Thermal Hotspot Detector...")
try:
    detector = ThermalHotspotDetector(clahe_clip=2.0, svm_kernel='rbf', egfs_features=20)
    print("[+] Detector initialized successfully")

    # Test preprocessing
    test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    preprocessed = detector.preprocess_image(test_image)
    assert preprocessed.shape == (256, 256), "Preprocessed image shape incorrect"
    print("[+] Image preprocessing working")

    # Test feature extraction
    features = detector.extract_all_features(preprocessed)
    assert len(features) > 0, "Feature vector empty"
    print(f"[+] Feature extraction working (extracted {len(features)} features)")

except Exception as e:
    print(f"[!] Detector initialization test failed: {e}")
    sys.exit(1)

# Test 5: Visualization
print("\n[5] Testing Visualization Module...")
try:
    viz = ThermalVisualization()
    print("[+] Visualization module initialized")

    # Test methods exist and are callable
    assert hasattr(viz, 'plot_feature_extraction_pipeline'), "Missing plot method"
    assert hasattr(viz, 'plot_egfs_results'), "Missing EGFS plot method"
    assert hasattr(viz, 'plot_confusion_matrix'), "Missing confusion matrix plot method"
    print("[+] All visualization methods available")

except Exception as e:
    print(f"[!] Visualization test failed: {e}")
    sys.exit(1)

# Test 6: Dataset availability
print("\n[6] Testing Dataset Availability...")
try:
    data_path = Path('/home/user/maskdatasetvgg16/data')
    test_path = Path('/home/user/maskdatasetvgg16/test')

    if data_path.exists():
        train_count = len(list(data_path.glob('*/*.jpg')))
        print(f"[+] Training images found: {train_count}")
    else:
        print("[!] Training data directory not found")

    if test_path.exists():
        test_count = len(list(test_path.glob('*/*.jpg')))
        print(f"[+] Test images found: {test_count}")
    else:
        print("[!] Test data directory not found")

except Exception as e:
    print(f"[!] Dataset check failed: {e}")

# Summary
print("\n[*] " + "="*70)
print("[+] ALL TESTS PASSED - System is ready for use!")
print("\n[*] Next step: Run 'python run_thermal_detection.py' for full pipeline execution")
print("\n    This will:")
print("    - Load and preprocess thermal images")
print("    - Extract LBP, LOOP, LDP features")
print("    - Apply EGFS feature selection")
print("    - Train SVM classifier")
print("    - Generate comprehensive visualizations")
print("\n" + "="*70 + "\n")
