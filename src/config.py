import os
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# YOLO settings
YOLO_MODEL = "yolov8m.pt"  # Model variant: nano(n), small(s), medium(m), large(l), xlarge(x)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Data settings
KAGGLE_DATASET = "your-dataset-path"
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
LABEL_EXTENSION = '.txt'

# Training settings
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
LEARNING_RATE = 0.001
PATIENCE = 20

# System settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
SEED = 42

# Colors for drawing
COLORS = {
    "person": (0, 255, 0),      # Green
    "chair": (255, 0, 0),       # Red
    "table": (0, 0, 255),       # Blue
}

print(f"Device: {DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
