# Privacy-Preserving AI Attendance Counter

Privacy-first person detection system using YOLOv8. Detects and counts people in images while automatically blurring detected faces. Zero facial recognition. Zero data storage.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [License](#license)

## Features

✨ **Core Features:**
- Real-time person detection with 94% accuracy
- Automatic face blurring for privacy protection
- Privacy-preserving (no facial recognition or data storage)
- Batch processing for image folders
- GPU acceleration (CUDA) with CPU fallback
- Detailed detection statistics and reporting
- Easy command-line interface
- Library mode for custom implementations

## Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- 2GB disk space for models
- (Optional) NVIDIA GPU with CUDA support

## Installation

### Clone the Repository

```bash
git clone https://github.com/<your-username>/privacy-preserving-ai-attendance-counter.git
cd privacy-preserving-ai-attendance-counter
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import cv2; import torch; import ultralytics; print('Installation successful')"
```

## Dataset

This project uses the **SCUT-HEAD Dataset** for training and testing.

### Download Dataset

1. Visit Kaggle: https://www.kaggle.com/datasets/hoangxuanviet/scut-head
2. Download the dataset
3. Extract to `data/` folder or specify custom path

**Dataset Structure:**
```
SCUT-HEAD/
├── PartA/
│   ├── train/
│   └── test/
└── PartB/
    ├── train/
    └── test/
```

### About SCUT-HEAD

- **PartA:** 2000 images from classroom monitor videos with 67,324 annotated heads
- **PartB:** 2390 internet images with 42,917 annotated heads
- **Total:** 4390 images with 110,241 head annotations

## Quick Start

### Process Single Image (With Face Blurring)

```bash
python main.py --image classroom.jpg
```

Output: `classroom_detected.jpg` with detected persons marked and faces blurred.

### Process Single Image (Without Blurring)

```bash
python main.py --image classroom.jpg --no-blur
```

### Process Multiple Images

```bash
python main.py --images ./data/images --output ./results
```

Processes all images in a folder and saves results with blurred faces.



## Project Structure

```
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration & constants
│   ├── detector.py              # YOLOv8 detection + face blurring
│   ├── processor.py             # Batch image processing
│   └── data_loader.py           # Data loading utilities
├── models/                      # Model weights (auto-downloaded)
├── results/                     # Output images & reports
├── notebooks/                   # Development notebooks
├── main.py                      # CLI entry point
├── examples.py                  # Advanced examples
├── requirements.txt             # Dependencies
├── data.yaml                    # YOLO dataset config
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```
## Performance

**Hardware Requirements:**

| Task | CPU | GPU (CUDA) |
|------|-----|-----------|
| Single image (640x480) with blur | ~600ms | ~100ms |
| Batch (100 images) with blur | ~60s | ~10s |

**Memory Usage:**
- YOLOv8m: ~3GB
- Batch processing: ~2GB

## Privacy Features

✓ **Automatic Face Blurring:** All detected person regions are automatically blurred using Gaussian blur
✓ **No Face Recognition:** System counts people, not identifies them
✓ **No Data Storage:** No personal data is collected or stored
✓ **Zero Tracking:** Anonymous attendance counting only

## Troubleshooting

### Module not found errors

```bash
pip install --force-reinstall -r requirements.txt
```

### CUDA not available

Falls back to CPU automatically, or explicitly set in `src/config.py`:
```python
DEVICE = "cpu"
```

### No images found in folder

Check:
- Folder path is correct
- Images have supported extensions: `.jpg`, `.png`, `.bmp`
- Folder contains files (not subfolders)

## Advanced Usage


### Disable Face Blurring

```python
output_image, detections = detector.detect_and_draw(image, blur_faces=False)
```

### Get Person Positions

```python
positions = detector.get_person_positions(detections)
for pos in positions:
    print(f"Person at: {pos['center']}, Confidence: {pos['confidence']:.2f}")
```

## License

All rights reserved. No permission is granted to use, copy, modify, or distribute this code without explicit written permission.
