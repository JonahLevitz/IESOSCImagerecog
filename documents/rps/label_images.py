"""
Script to help label images for training.
Uses OpenCV to draw bounding boxes and save YOLO format annotations.
"""
import cv2
import os
from pathlib import Path
import numpy as np

class_names = ['rock', 'paper', 'scissors']
image_dir = Path('dataset/train/images')
label_dir = Path('dataset/train/labels')
label_dir.mkdir(parents=True, exist_ok=True)

class BoundingBoxDrawer:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.original_image = image.copy()
        self.image = image.copy()
        self.points = []
        self.drawing = False
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.moveWindow(self.window_name, 100, 100)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing
            self.drawing = True
            self.points = [(x, y)]
            self.image = self.original_image.copy()
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw temporary rectangle
            temp_image = self.original_image.copy()
            cv2.rectangle(temp_image, self.points[0], (x, y), (0, 255, 0), 2)
            self.image = temp_image
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # Finish drawing
            self.drawing = False
            self.points.append((x, y))
            cv2.rectangle(self.image, self.points[0], self.points[1], (0, 255, 0), 2)
    
    def get_bbox(self):
        """Get the bounding box coordinates."""
        if len(self.points) == 2:
            return self.points
        return None
    
    def reset(self):
        """Reset the drawing."""
        self.points = []
        self.image = self.original_image.copy()
        self.drawing = False

def get_gesture_type():
    """Ask user to specify the gesture type."""
    while True:
        print("\nWhat gesture is shown in this image?")
        print("0: rock")
        print("1: paper")
        print("2: scissors")
        print("s: skip (no gesture/unclear)")
        print("q: quit")
        choice = input("Enter choice: ").strip().lower()
        
        if choice == 'q':
            return None
        elif choice == 's':
            return 'skip'
        try:
            choice = int(choice)
            if 0 <= choice < len(class_names):
                return choice
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def label_image(image_path):
    """Label one image with a bounding box."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return True
    
    height, width = img.shape[:2]
    
    # Create bounding box drawer
    drawer = BoundingBoxDrawer('Label Image', img)
    
    print("\nInstructions:")
    print("1. Click and drag to draw bounding box")
    print("2. Press 'r' to reset if you make a mistake")
    print("3. Press 'space' to save and continue")
    print("4. Press 'q' to quit")
    print("5. Type 's' when prompted to skip images with no clear gesture")
    print("\nDrawing bounding box for:", image_path.name)
    
    while True:
        # Show image
        cv2.imshow('Label Image', drawer.image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):  # Reset
            drawer.reset()
        
        elif key == ord(' '):  # Save
            # Get gesture type from user
            class_idx = get_gesture_type()
            if class_idx is None:  # User quit
                cv2.destroyAllWindows()
                return False
            elif class_idx == 'skip':  # Skip this image
                print(f"Skipping {image_path.name}")
                break
            
            bbox = drawer.get_bbox()
            if bbox:
                # Convert to YOLO format (x_center, y_center, width, height)
                x1, y1 = min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1])
                x2, y2 = max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1])
                
                x_center = (x1 + x2) / (2 * width)
                y_center = (y1 + y2) / (2 * height)
                w = abs(x2 - x1) / width
                h = abs(y2 - y1) / height
                
                # Save label
                label_path = label_dir / (image_path.stem + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")
                print(f"Saved label to {label_path}")
                break
            else:
                print("Please draw a bounding box first")
        
        elif key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            return False
    
    cv2.destroyAllWindows()
    return True

def main():
    """Main function for labeling images."""
    print("Starting image labeling process")
    print("\nClass indices:")
    for idx, name in enumerate(class_names):
        print(f"{idx}: {name}")
    
    # Get all images
    images = sorted(image_dir.glob('*.jpg'))
    if not images:
        print(f"\nNo images found in {image_dir}")
        return
    
    print(f"\nFound {len(images)} images to label")
    input("Press Enter to start labeling...")
    
    # Label all images
    for image_path in images:
        # Skip if label already exists
        label_path = label_dir / (image_path.stem + '.txt')
        if label_path.exists():
            print(f"Skipping {image_path.name} - label already exists")
            continue
            
        print(f"\nLabeling {image_path.name}")
        
        if not label_image(image_path):
            print("\nLabeling interrupted by user")
            break
    
    print("\nLabeling complete!")

if __name__ == "__main__":
    main() 