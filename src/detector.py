import cv2
import numpy as np
from ultralytics import YOLO
from .config import YOLO_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, DEVICE


class AttendanceDetector:
    """
    Attendance detection class using YOLOv8
    """
    
    def __init__(self, model_name=YOLO_MODEL):
        """
        Initialize YOLO model
        
        Args:
            model_name (str): Name of YOLO model to use
        """
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        print(f"YOLO model loaded: {model_name}")
        print(f"Available classes: {self.class_names}")
    
    def detect(self, image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD):
        """
        Detect persons in image
        
        Args:
            image (numpy.ndarray): Input image
            conf (float): Confidence threshold for detection
            iou (float): IOU threshold for NMS
            
        Returns:
            dict: Detection results
        """
        results = self.model(image, conf=conf, iou=iou, device=DEVICE)
        detections = {
            'persons': [],
            'other_objects': [],
            'total_detections': 0
        }
        
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls),
                        'class_name': self.class_names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': {
                            'x_min': int(box.xyxy[0][0]),
                            'y_min': int(box.xyxy[0][1]),
                            'x_max': int(box.xyxy[0][2]),
                            'y_max': int(box.xyxy[0][3]),
                        }
                    }
                    
                    # Separate persons from other objects
                    if detection['class_name'] == 'person':
                        detections['persons'].append(detection)
                    else:
                        detections['other_objects'].append(detection)
                    
                    detections['total_detections'] += 1
        
        return detections
    
    def detect_and_draw(self, image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, blur_faces=True):
        """
        Detect persons and draw boxes on image. Optionally blur detected faces.
        
        Args:
            image (numpy.ndarray): Input image
            conf (float): Confidence threshold for detection
            iou (float): IOU threshold for NMS
            blur_faces (bool): Whether to blur detected person faces
            
        Returns:
            tuple: Modified image and detection results
        """
        detections = self.detect(image, conf=conf, iou=iou)
        output_image = image.copy()
        
        # Draw boxes for persons in green and blur faces if enabled
        for person in detections['persons']:
            bbox = person['bbox']
            
            # Blur face region if enabled
            if blur_faces:
                x_min = max(0, bbox['x_min'])
                y_min = max(0, bbox['y_min'])
                x_max = min(output_image.shape[1], bbox['x_max'])
                y_max = min(output_image.shape[0], bbox['y_max'])
                
                face_region = output_image[y_min:y_max, x_min:x_max]
                blurred = cv2.GaussianBlur(face_region, (51, 51), 0)
                output_image[y_min:y_max, x_min:x_max] = blurred
            
            # Draw bounding box
            cv2.rectangle(output_image, 
                         (bbox['x_min'], bbox['y_min']),
                         (bbox['x_max'], bbox['y_max']),
                         (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Person {person['confidence']:.2f}"
            cv2.putText(output_image, label,
                       (bbox['x_min'], bbox['y_min'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw boxes for other objects in red
        for obj in detections['other_objects']:
            bbox = obj['bbox']
            cv2.rectangle(output_image,
                         (bbox['x_min'], bbox['y_min']),
                         (bbox['x_max'], bbox['y_max']),
                         (0, 0, 255), 1)
        
        return output_image, detections
    
    def count_persons(self, detections):
        """
        Count number of detected persons
        
        Args:
            detections (dict): Detection results
            
        Returns:
            int: Number of persons
        """
        return len(detections['persons'])
    
    def get_person_positions(self, detections):
        """
        Get positions of detected persons
        
        Args:
            detections (dict): Detection results
            
        Returns:
            list: List of person positions
        """
        positions = []
        for person in detections['persons']:
            bbox = person['bbox']
            center_x = (bbox['x_min'] + bbox['x_max']) // 2
            center_y = (bbox['y_min'] + bbox['y_max']) // 2
            positions.append({
                'center': (center_x, center_y),
                'bbox': bbox,
                'confidence': person['confidence']
            })
        return positions
    
    def get_model_info(self):
        """
        Get model information
        
        Returns:
            dict: Model information
        """
        return {
            'model': str(self.model),
            'task': self.model.task,
            'classes': self.class_names,
            'num_classes': len(self.class_names)
        }


if __name__ == "__main__":
    print("Attendance Detection Module")
    print("Use this module to detect persons in images")
