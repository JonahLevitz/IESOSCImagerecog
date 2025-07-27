"""
Script to collect training data for rock, paper, scissors detection.
"""
import cv2
import os
import time
from pathlib import Path
import numpy as np

def collect_gesture_images(gesture_name, start_index, num_images=30):
    """
    Collect images for a specific gesture.
    
    Args:
        gesture_name (str): Name of the gesture to collect
        start_index (int): Starting index for image numbering
        num_images (int): Number of images to collect
    """
    base_dir = Path('dataset/train/images')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        print(f"\nCollecting images for {gesture_name}")
        print(f"Please make a {gesture_name} gesture when prompted")
        print(f"Will capture {num_images} images")
        
        for i in range(num_images):
            print(f"\nCapturing image {i+1}/{num_images}")
            print("Get ready...")
            
            # Countdown
            for j in range(3, 0, -1):
                _, frame = cap.read()
                # Display countdown
                cv2.putText(frame, str(j), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 0, 0), 2)
                cv2.imshow('Capture', frame)
                cv2.waitKey(1)
                time.sleep(1)
            
            # Capture image
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame")
                continue
            
            # Save image
            filename = f"{gesture_name}_{start_index + i:03d}.jpg"
            filepath = base_dir / filename
            cv2.imwrite(str(filepath), frame)
            print(f"Saved {filename}")
            
            # Show captured frame
            cv2.putText(frame, "Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.imshow('Capture', frame)
            
            # Wait before next capture
            time.sleep(2)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCapture interrupted by user")
                return False
        
        return True
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function."""
    print("Starting data collection for Rock Paper Scissors detector")
    print("\nInstructions:")
    print("1. You will be prompted to make specific gestures")
    print("2. For each gesture, multiple images will be captured")
    print("3. A countdown will show when each image will be captured")
    print("4. Press 'q' at any time to quit")
    print("\nMake sure you:")
    print("- Have good lighting")
    print("- Show clear gestures")
    print("- Vary your hand position and angle slightly")
    
    # Collect remaining paper images
    print("\nCollecting remaining paper images...")
    if not collect_gesture_images('paper', 20, 35):  # Need 35 more paper images
        return
    
    # Collect scissors images
    print("\nCollecting scissors images...")
    if not collect_gesture_images('scissors', 0, 55):  # Need all 55 scissors images
        return
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main() 