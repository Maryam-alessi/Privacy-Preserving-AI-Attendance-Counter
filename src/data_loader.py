import os
import glob
import cv2
import time
from pathlib import Path
import kagglehub
from .config import DATA_DIR, IMG_EXTENSIONS, LABEL_EXTENSION


def download_with_retry(dataset, retries=5, wait=10):
    """
    Download a Kaggle dataset with retry on failure
    
    Args:
        dataset (str): Kaggle dataset path
        retries (int): Number of retry attempts
        wait (int): Wait time between retries in seconds
        
    Returns:
        str: Path to downloaded dataset
    """
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1}/{retries}: Downloading {dataset}...")
            path = kagglehub.dataset_download(dataset)
            print(f"Successfully downloaded: {path}")
            return path
        except Exception as e:
            print(f"Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
    
    raise Exception(f"Failed to download dataset after {retries} attempts")


def load_images(path, extensions=IMG_EXTENSIONS):
    """
    Load all images from a folder
    
    Args:
        path (str): Folder path
        extensions (list): Supported image extensions
        
    Returns:
        list: List of image paths
    """
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))
    
    print(f"Found {len(images)} images")
    return sorted(images)


def load_labels(path):
    """
    Load YOLO format label files (.txt)
    
    Args:
        path (str): Folder path
        
    Returns:
        list: List of label file paths
    """
    labels_path = os.path.join(path, "**", "*" + LABEL_EXTENSION)
    txt_files = glob.glob(labels_path, recursive=True)
    
    print(f"Found {len(txt_files)} label files")
    return sorted(txt_files)


def load_label_content(label_file):
    """
    Read label file content
    
    Args:
        label_file (str): Path to label file
        
    Returns:
        list: List of annotations (bounding boxes)
    """
    annotations = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # class_id x_center y_center width height
                    annotations.append({
                        'class': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
    
    return annotations


def image_to_yolo_bbox(image_path, annotations):
    """
    Convert YOLO annotations to actual image coordinates
    
    Args:
        image_path (str): Path to image
        annotations (list): List of annotations
        
    Returns:
        tuple: Image and updated boxes
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, []
    
    height, width = image.shape[:2]
    boxes = []
    
    for ann in annotations:
        # Convert from YOLO format to pixel coordinates
        x_center = int(ann['x_center'] * width)
        y_center = int(ann['y_center'] * height)
        w = int(ann['width'] * width)
        h = int(ann['height'] * height)
        
        x_min = max(0, x_center - w // 2)
        y_min = max(0, y_center - h // 2)
        x_max = min(width, x_center + w // 2)
        y_max = min(height, y_center + h // 2)
        
        boxes.append({
            'class': ann['class'],
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'x_center': x_center,
            'y_center': y_center
        })
    
    return image, boxes


if __name__ == "__main__":
    print("Data Loading Module")
    print("Use this module to load data from various sources")
