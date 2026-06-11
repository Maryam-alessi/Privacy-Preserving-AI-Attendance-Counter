#!/usr/bin/env python3
"""
Advanced usage examples for Privacy-Preserving AI Attendance Counter.
"""

import cv2
import sys
import os

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import AttendanceDetector
from src.processor import AttendanceProcessor


# ---------------------------------------------------------------------------
# Example 1: Custom detection and drawing on an image
# ---------------------------------------------------------------------------
def example_1_custom_detection():
    """Detect persons and draw custom annotations."""
    print("\n" + "=" * 60)
    print("Example 1: Custom detection and drawing")
    print("=" * 60)

    test_image = cv2.imread("test_image.jpg")
    if test_image is None:
        print("test_image.jpg not found")
        return

    detector = AttendanceDetector()
    detections = detector.detect(test_image, conf=0.45)

    output_image = test_image.copy()

    for person in detections['persons']:
        bbox = person['bbox']

        # Thick bounding box
        cv2.rectangle(output_image,
                      (bbox['x_min'], bbox['y_min']),
                      (bbox['x_max'], bbox['y_max']),
                      (0, 255, 0), 3)

        # Center dot
        center_x = (bbox['x_min'] + bbox['x_max']) // 2
        center_y = (bbox['y_min'] + bbox['y_max']) // 2
        cv2.circle(output_image, (center_x, center_y), 10, (255, 0, 0), -1)

        # Confidence label
        label = f"Person {person['confidence']:.1%}"
        cv2.putText(output_image, label,
                    (bbox['x_min'], bbox['y_min'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(output_image,
                f"Persons: {detector.count_persons(detections)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imwrite("example1_result.jpg", output_image)
    print("Saved: example1_result.jpg")


# ---------------------------------------------------------------------------
# Example 2: Compare different confidence thresholds
# ---------------------------------------------------------------------------
def example_2_confidence_comparison():
    """Compare detection results across confidence levels."""
    print("\n" + "=" * 60)
    print("Example 2: Confidence threshold comparison")
    print("=" * 60)

    test_image = cv2.imread("test_image.jpg")
    if test_image is None:
        print("test_image.jpg not found")
        return

    detector = AttendanceDetector()
    confidence_levels = [0.3, 0.5, 0.7, 0.9]

    print("Image: test_image.jpg\n")
    print(f"{'Confidence':<15} {'Persons':<15} {'Total objects':<15}")
    print("-" * 45)

    for conf in confidence_levels:
        detections = detector.detect(test_image, conf=conf)
        person_count = detector.count_persons(detections)
        total_count = detections['total_detections']
        print(f"{conf:<15.1f} {person_count:<15} {total_count:<15}")

    print("\nNote: higher confidence means fewer false detections")


# ---------------------------------------------------------------------------
# Example 3: Batch process a folder of images
# ---------------------------------------------------------------------------
def example_3_batch_process_images():
    """Process multiple images from a folder."""
    print("\n" + "=" * 60)
    print("Example 3: Batch process images folder")
    print("=" * 60)

    folder_path = "images_folder"
    output_folder = "results_folder"

    if not os.path.exists(folder_path):
        print(f"{folder_path} not found")
        return

    processor = AttendanceProcessor()
    results = processor.process_images_batch(folder_path, output_dir=output_folder)
    processor.print_statistics(results)


# ---------------------------------------------------------------------------
# Example 4: Detailed detection report
# ---------------------------------------------------------------------------
def example_4_detailed_report():
    """Generate a detailed text report of detections."""
    print("\n" + "=" * 60)
    print("Example 4: Detailed detection report")
    print("=" * 60)

    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        print(f"{image_path} not found")
        return

    image = cv2.imread(image_path)
    detector = AttendanceDetector()
    detections = detector.detect(image)

    report = []
    report.append("=" * 60)
    report.append("Detection Report")
    report.append("=" * 60)
    report.append(f"Image: {image_path}")
    report.append(f"Image size: {image.shape[1]}x{image.shape[0]}")
    report.append("")
    report.append("Detected persons:")
    report.append(f"  Total: {len(detections['persons'])}")
    report.append("")

    for i, person in enumerate(detections['persons'], 1):
        bbox = person['bbox']
        width = bbox['x_max'] - bbox['x_min']
        height = bbox['y_max'] - bbox['y_min']
        area = width * height

        report.append(f"  Person {i}:")
        report.append(f"    - Confidence: {person['confidence']:.2%}")
        report.append(f"    - Position: ({bbox['x_min']}, {bbox['y_min']})")
        report.append(f"    - Size: {width}x{height}")
        report.append(f"    - Area: {area} px")
        report.append("")

    report.append("Other objects:")
    report.append(f"  Count: {len(detections['other_objects'])}")

    report_text = "\n".join(report)
    print(report_text)

    with open("detection_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nSaved: detection_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("Privacy-Preserving AI Attendance Counter - Examples")
    print("=" * 60)

    examples = {
        1: ("Custom detection and drawing", example_1_custom_detection),
        2: ("Confidence threshold comparison", example_2_confidence_comparison),
        3: ("Batch process images folder", example_3_batch_process_images),
        4: ("Detailed detection report", example_4_detailed_report),
    }

    print("\nSelect an example to run:")
    for num, (name, _) in examples.items():
        print(f"  {num}. {name}")
    print("  5. Run all examples")

    try:
        choice = int(input("\nYour choice (1-5): "))

        if choice == 5:
            for num, (name, func) in examples.items():
                try:
                    func()
                except Exception as e:
                    print(f"Error in example {num}: {e}")
        elif choice in examples:
            examples[choice][1]()
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
