"""
Run the emergency detection system using webcam or video input.
"""
import cv2
import argparse
from pathlib import Path
from core import EmergencyDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Emergency Detection System')
    parser.add_argument('--source', type=str, default='0',
                       help='Source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    args = parser.parse_args()
    
    # Initialize detector
    detector = EmergencyDetector(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    # Initialize video capture
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    print("\nEmergency Detection System")
    print("-------------------------")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Source: {'Webcam' if args.source == '0' else args.source}")
    print("\nPress 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame, detections = detector.process_frame(frame)
            
            # Print detections
            for det in detections:
                print(f"\nDetected {det['type'].upper()}")
                print(f"Class: {det['class_name']}")
                print(f"Confidence: {det['confidence']:.3f}")
            
            # Show frame
            cv2.imshow('Emergency Detection', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetection system shutdown complete")

if __name__ == '__main__':
    main() 