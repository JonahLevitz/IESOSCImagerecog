"""
Test script to debug model detection.
"""
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    """Test detection with debug information."""
    # Load model
    print("Loading model...")
    model = YOLO('runs/train/rps_detector/weights/best.pt')
    print(f"Model loaded. Classes: {model.names}")
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nStarting detection test...")
    print("Press 'q' to quit, 'c' to capture and analyze a frame")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Show frame
            cv2.imshow('Test Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\nAnalyzing frame...")
                # Run detection with different confidence thresholds
                for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    print(f"\nConfidence threshold: {conf}")
                    results = model(frame, conf=conf, verbose=False)
                    
                    if len(results) > 0:
                        result = results[0]
                        print(f"Number of detections: {len(result.boxes)}")
                        
                        for i, box in enumerate(result.boxes):
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            print(f"Detection {i+1}: {class_name} ({confidence:.3f})")
                    else:
                        print("No detections")
                
                print("\nPress 'c' to analyze another frame or 'q' to quit")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 