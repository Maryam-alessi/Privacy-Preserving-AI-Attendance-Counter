import cv2
import os
from pathlib import Path
from .detector import AttendanceDetector
from .data_loader import load_images


class AttendanceProcessor:
    """
    Process attendance on images and folders
    """
    
    def __init__(self):
        """Initialize attendance processor"""
        self.detector = AttendanceDetector()
        self.attendance_records = []
    
    def process_images_batch(self, image_dir, output_dir=None, blur_faces=True):
        """
        Process batch of images with optional face blurring
        
        Args:
            image_dir (str): Images folder
            output_dir (str): Output folder (optional)
            blur_faces (bool): Whether to blur detected faces
            
        Returns:
            dict: Processing results
        """
        images = load_images(image_dir)
        
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nProcessing {len(images)} images from {image_dir}")
        if blur_faces:
            print("Face blurring: ENABLED")
        
        os.makedirs(output_dir, exist_ok=True) if output_dir else None
        
        results = {
            'total_images': len(images),
            'images_with_persons': 0,
            'total_persons': 0,
            'max_persons_in_image': 0,
            'detections': []
        }
        
        for i, image_path in enumerate(images):
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            output_image, detections = self.detector.detect_and_draw(image, blur_faces=blur_faces)
            num_persons = self.detector.count_persons(detections)
            
            if num_persons > 0:
                results['images_with_persons'] += 1
                results['total_persons'] += num_persons
                results['max_persons_in_image'] = max(results['max_persons_in_image'], num_persons)
            
            results['detections'].append({
                'image': image_path,
                'persons_count': num_persons,
                'total_objects': detections['total_detections']
            })
            
            # Save output image if needed
            if output_dir:
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, output_image)
            
            # Progress message
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(images)} images processed")
        
        return results
    
    def print_statistics(self, results):
        """
        Print processing statistics
        
        Args:
            results (dict): Processing results
        """
        print("\n" + "="*50)
        print("Processing Statistics")
        print("="*50)
        
        if 'total_images' in results:
            print(f"Total images: {results['total_images']}")
            print(f"Images with persons: {results['images_with_persons']}")
            print(f"Total persons detected: {results['total_persons']}")
            print(f"Max persons in single image: {results['max_persons_in_image']}")
        
        print("="*50)


if __name__ == "__main__":
    print("Attendance Processing Module")
    print("Use this module to process images")
