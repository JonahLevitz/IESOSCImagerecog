"""
Simplified script to collect and label training data for rock, paper, scissors detection.
"""
import cv2
import os
from pathlib import Path
import time

# Setup directories
dataset_dir = Path('new_dataset')
image_dir = dataset_dir / 'train' / 'images'
label_dir = dataset_dir / 'train' / 'labels'
image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

class DataCollector:
    def __init__(self):
        self.drawing = False
        self.point1 = None
        self.point2 = None
        self.current_image = None
        self.current_frame = None
        self.current_gesture = None
        self.counters = {'rock': 0, 'paper': 0, 'scissors': 0}
    
    def draw_box(self, event, x, y, flags, param):
        """Mouse callback function for drawing bounding box."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.point1 = (x, y)
            self.point2 = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_image = self.current_frame.copy()
                cv2.rectangle(self.current_image, self.point1, (x, y), (0, 255, 0), 2)
                cv2.imshow('Collect Data', self.current_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.point2 = (x, y)
                self.drawing = False
                cv2.rectangle(self.current_image, self.point1, self.point2, (0, 255, 0), 2)
                cv2.imshow('Collect Data', self.current_image)
    
    def run(self):
        """Main data collection loop."""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('Collect Data')
        cv2.setMouseCallback('Collect Data', lambda *args: self.draw_box(*args))
        
        print("\nData Collection Instructions:")
        print("1. Press 'r' for rock, 'p' for paper, 's' for scissors")
        print("2. Draw a box around your hand by clicking and dragging")
        print("3. Press SPACE to save the image and label")
        print("4. Press 'q' to quit")
        print("\nTips:")
        print("- Make sure your hand is clearly visible")
        print("- Vary your hand position and angle")
        print("- Include some background variation")
        print("\nCurrent counts:")
        for gesture, count in self.counters.items():
            print(f"{gesture}: {count}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # If no box is being drawn, update the display
            if not self.drawing:
                self.current_image = self.current_frame.copy()
                if self.point1 and self.point2:
                    cv2.rectangle(self.current_image, self.point1, self.point2, (0, 255, 0), 2)
            
            # Show current gesture if selected
            if self.current_gesture:
                cv2.putText(self.current_image, f"Current: {self.current_gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Collect Data', self.current_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('r'), ord('p'), ord('s')]:
                # Set current gesture
                self.current_gesture = {'r': 'rock', 'p': 'paper', 's': 'scissors'}[chr(key)]
                self.point1 = None
                self.point2 = None
                print(f"\nSelected: {self.current_gesture}")
                print("Now draw a box around your hand")
            
            elif key == ord(' '):
                # Save image and label if we have a gesture and box
                if self.current_gesture and self.point1 and self.point2:
                    # Get coordinates
                    x1, y1 = min(self.point1[0], self.point2[0]), min(self.point1[1], self.point2[1])
                    x2, y2 = max(self.point1[0], self.point2[0]), max(self.point1[1], self.point2[1])
                    
                    # Convert to YOLO format
                    width = self.current_frame.shape[1]
                    height = self.current_frame.shape[0]
                    x_center = (x1 + x2) / (2 * width)
                    y_center = (y1 + y2) / (2 * height)
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Get class index
                    class_idx = {'rock': 0, 'paper': 1, 'scissors': 2}[self.current_gesture]
                    
                    # Save image and label
                    image_path = image_dir / f"{self.current_gesture}_{self.counters[self.current_gesture]:03d}.jpg"
                    label_path = label_dir / f"{self.current_gesture}_{self.counters[self.current_gesture]:03d}.txt"
                    
                    cv2.imwrite(str(image_path), self.current_frame)
                    with open(label_path, 'w') as f:
                        f.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")
                    
                    self.counters[self.current_gesture] += 1
                    print(f"\nSaved {image_path}")
                    print("\nCurrent counts:")
                    for gesture, count in self.counters.items():
                        print(f"{gesture}: {count}")
                    
                    # Reset for next capture
                    self.current_gesture = None
                    self.point1 = None
                    self.point2 = None
                else:
                    if not self.current_gesture:
                        print("\nPlease select a gesture first (r/p/s)")
                    if not (self.point1 and self.point2):
                        print("\nPlease draw a box around your hand")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nData collection complete!")
        print("Final counts:")
        for gesture, count in self.counters.items():
            print(f"{gesture}: {count}")

def main():
    collector = DataCollector()
    collector.run()

if __name__ == "__main__":
    main() 