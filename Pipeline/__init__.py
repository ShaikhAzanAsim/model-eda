"""
EDA Pipeline - Data Analysis for Image Datasets

A comprehensive PyQt5-based framework for exploring and analyzing image datasets
with automatic detection of dataset types and intelligent annotation visualization.

Main modules:
- pyqt: GUI application
- dataset_detector: Auto-detect dataset type and structure
- annotation_handlers: Factory pattern for annotation visualization
- video_generator: Create slideshow videos
"""

__version__ = "1.0.0"
__author__ = "Data Analysis Team"

# Conditional imports - only import if modules exist
try:
    from dataset_detector import DatasetDetector, DatasetInfo
except ImportError:
    pass

try:
    from annotation_handlers import (
        AnnotationHandler,
        AnnotationHandlerFactory,
        ClassificationHandler,
        SegmentationHandler,
        DetectionHandler,
        PoseHandler,
        UnknownHandler,
    )
except ImportError:
    pass

try:
    from video_generator import VideoGenerator
except ImportError:
    pass

__all__ = [
    'DatasetDetector',
    'DatasetInfo',
    'AnnotationHandler',
    'AnnotationHandlerFactory',
    'ClassificationHandler',
    'SegmentationHandler',
    'DetectionHandler',
    'PoseHandler',
    'UnknownHandler',
    'VideoGenerator',
]
