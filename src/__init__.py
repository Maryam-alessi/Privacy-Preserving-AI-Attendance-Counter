"""
Privacy-Preserving AI Attendance Counter
Privacy-first attendance detection using YOLOv8 computer vision.
"""

from .config import *
from .data_loader import load_images, load_labels, load_label_content
from .detector import AttendanceDetector
from .processor import AttendanceProcessor

__version__ = "1.0.0"

__all__ = [
    'AttendanceDetector',
    'AttendanceProcessor',
    'load_images',
    'load_labels',
    'load_label_content',
]
