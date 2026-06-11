#!/usr/bin/env python3
"""
Privacy-Preserving AI Attendance Counter - Main Program
"""

import os
import sys
import argparse
from pathlib import Path

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import AttendanceDetector
from src.processor import AttendanceProcessor
from src.data_loader import load_images
import cv2


def process_single_image(image_path, blur_faces=True):
    """
    Process single image and detect persons
    
    Args:
        image_path (str): Path to image
        blur_faces (bool): Whether to blur detected faces
    """
    print(f"\nProcessing image: {image_path}")
    if blur_faces:
        print("Face blurring: ENABLED")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
    
    # Detect
    detector = AttendanceDetector()
    output_image, detections = detector.detect_and_draw(image, blur_faces=blur_faces)
    
    # Print results
    print(f"Detection completed")
    print(f"  - Number of persons: {detector.count_persons(detections)}")
    print(f"  - Total objects: {detections['total_detections']}")
    
    # Save image
    output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
    cv2.imwrite(output_path, output_image)
    print(f"Image saved: {output_path}")


def process_images_folder(folder_path, output_folder=None, blur_faces=True):
    """
    Process folder of images
    
    Args:
        folder_path (str): Path to folder
        output_folder (str): Path to output folder
        blur_faces (bool): Whether to blur detected faces
    """
    print(f"\nProcessing images folder: {folder_path}")
    if blur_faces:
        print("Face blurring: ENABLED")
    
    processor = AttendanceProcessor()
    results = processor.process_images_batch(folder_path, output_dir=output_folder, blur_faces=blur_faces)
    
    processor.print_statistics(results)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving AI Attendance Counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Process single image with face blurring
  python main.py --image path/to/image.jpg
  
  # Process single image without blurring
  python main.py --image path/to/image.jpg --no-blur
  
  # Process images folder with blurring
  python main.py --images path/to/folder --output path/to/output
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to image for processing')
    parser.add_argument('--images', type=str, help='Path to images folder')
    parser.add_argument('--output', type=str, help='Path to output file/folder')
    parser.add_argument('--no-blur', action='store_true', help='Disable face blurring (default: enabled)')
    
    args = parser.parse_args()
    
    # Check if at least one argument provided
    if not any([args.image, args.images]):
        parser.print_help()
        return
    
    blur_faces = not args.no_blur
    
    # Process image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        process_single_image(args.image, blur_faces=blur_faces)
    
    # Process images folder
    elif args.images:
        if not os.path.exists(args.images):
            print(f"Folder not found: {args.images}")
            return
        process_images_folder(args.images, output_folder=args.output, blur_faces=blur_faces)


if __name__ == "__main__":
    print("="*60)
    print("Privacy-Preserving AI Attendance Counter")
    print("AI3302 - Computer Vision Project")
    print("="*60)
    
    main()
