"""
Script to rename captured images to the correct format for training.
"""
import os
from pathlib import Path

def rename_images():
    """Rename images to match the expected format."""
    image_dir = Path('dataset/train/images')
    
    # Get all images
    images = sorted(image_dir.glob('*.jpg'))
    total_images = len(images)
    
    # Calculate images per class
    images_per_class = total_images // 3
    
    # Count for each class
    counters = {'rock': 0, 'paper': 0, 'scissors': 0}
    
    print(f"Found {total_images} images")
    print(f"Will use {images_per_class} images per class")
    
    print("\nRenaming images...")
    for i, img_path in enumerate(images):
        # Skip if already in correct format
        if any(img_path.stem.startswith(cls) for cls in counters.keys()):
            continue
            
        # Determine class based on position in sequence
        if i < images_per_class:
            cls = 'rock'
        elif i < images_per_class * 2:
            cls = 'paper'
        else:
            cls = 'scissors'
        
        # Create new filename
        counter = counters[cls]
        if counter >= images_per_class:
            continue  # Skip extra images
            
        new_name = f"{cls}_{counter:03d}.jpg"
        new_path = img_path.parent / new_name
        
        # Rename file
        img_path.rename(new_path)
        print(f"Renamed {img_path.name} to {new_name}")
        
        # Update counter
        counters[cls] += 1
    
    print("\nRename complete!")
    print("Images per class:")
    for cls, count in counters.items():
        print(f"{cls}: {count}")

if __name__ == "__main__":
    rename_images() 