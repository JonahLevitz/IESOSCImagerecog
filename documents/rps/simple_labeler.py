"""
Simplified script for labeling images.
"""
import cv2
import os
from pathlib import Path

# Setup paths
image_dir = Path('dataset/train/images')
label_dir = Path('dataset/train/labels')
label_dir.mkdir(parents=True, exist_ok=True)

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
box_drawn = False
current_image = None
current_box = None

def draw_box(event, x, y, flags, param):
    """Mouse callback function."""
    global drawing, ix, iy, box_drawn, current_image, current_box
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box_drawn = False
        current_box = None
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', img_copy)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box_drawn = True
        current_box = (ix, iy, x, y)
        cv2.rectangle(current_image, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', current_image)

def main():
    global current_image, current_box
    
    # Create window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_box)
    
    # Get list of images
    images = sorted(list(image_dir.glob('*.jpg')))
    if not images:
        print("No images found!")
        return
    
    print("\nInstructions:")
    print("1. Draw box by clicking and dragging")
    print("2. Press:")
    print("   0 - for rock")
    print("   1 - for paper")
    print("   2 - for scissors")
    print("   s - to skip image")
    print("   r - to redraw box")
    print("   q - to quit")
    
    for img_path in images:
        # Skip if already labeled
        label_path = label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            continue
        
        # Read and display image
        current_image = cv2.imread(str(img_path))
        if current_image is None:
            print(f"Could not read {img_path}")
            continue
            
        height, width = current_image.shape[:2]
        current_box = None
        box_drawn = False
        
        print(f"\nLabeling: {img_path.name}")
        cv2.imshow('Image', current_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                return
                
            elif key == ord('r'):
                current_image = cv2.imread(str(img_path))
                current_box = None
                box_drawn = False
                cv2.imshow('Image', current_image)
                
            elif key == ord('s'):
                print("Skipping image")
                break
                
            elif key in [ord('0'), ord('1'), ord('2')]:
                if not current_box:
                    print("Please draw a box first")
                    continue
                    
                # Get coordinates
                x1, y1, x2, y2 = current_box
                # Convert to YOLO format
                x_center = (min(x1, x2) + max(x1, x2)) / (2 * width)
                y_center = (min(y1, y2) + max(y1, y2)) / (2 * height)
                w = abs(x2 - x1) / width
                h = abs(y2 - y1) / height
                
                # Save label
                class_idx = int(chr(key))
                with open(label_path, 'w') as f:
                    f.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")
                print(f"Saved label for {img_path.name}")
                break
    
    cv2.destroyAllWindows()
    print("\nLabeling complete!")

if __name__ == "__main__":
    main() 