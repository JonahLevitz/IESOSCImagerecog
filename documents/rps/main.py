"""
Main script for running the Rock Paper Scissors detection system.
"""
import cv2
from utils.video import VideoCapture
from utils.detector import HandGestureDetector
from utils.logger import DetectionLogger

def main():
    """Main function to run the detection system."""
    # Initialize components
    detector = HandGestureDetector()
    logger = DetectionLogger()
    
    print("Initializing Rock Paper Scissors Detection System...")
    print(f"Model info: {detector.get_model_info()}")
    print("Press 'q' to quit")
    
    # Use context manager for video capture
    with VideoCapture() as video:
        while True:
            # Read frame
            success, frame = video.read_frame()
            if not success:
                print("Error: Could not read frame")
                break
            
            # Detect hand gesture
            label, confidence, bbox = detector.detect(frame)
            
            # If detection found, process it
            if label and confidence > 0:
                # Draw detection on frame
                frame = video.draw_detection(frame, label, confidence, bbox)
                
                # Save detection
                image_path = logger.save_detection(frame, label, confidence)
                print(f"Detection saved: {label} ({confidence:.2f}) - {image_path}")
            
            # Display frame
            video.show_frame(frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("System shutdown complete")

if __name__ == "__main__":
    main() 