"""
Real-Time Hotspot Classification Pipeline
==========================================

Optimized processing chain for hotspot detection in thermal images.
Target: ~30ms per frame with 94-97% accuracy.
"""

__version__ = "1.0.0"

from .preprocessing.preprocessor import ThermalPreprocessor
from .feature_extraction.extractors import FastFeatureExtractor
from .feature_selection.egfs import EGFSSelector
from .multi_scale.mrtp import MRTPAnalyzer
from .classification.classifier import HotspotClassifier
from .pipeline import RealTimeHotspotPipeline

__all__ = [
    'ThermalPreprocessor',
    'FastFeatureExtractor',
    'EGFSSelector',
    'MRTPAnalyzer',
    'HotspotClassifier',
    'RealTimeHotspotPipeline'
]
