"""
Script to train YOLOv8 model on rock, paper, scissors dataset.
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import random
import time

def prepare_validation_set():
    """
    Split dataset into training and validation sets.
    Moves 20% of images and their labels to validation set.
    """
    train_img_dir = Path('dataset/train/images')
    train_label_dir = Path('dataset/train/labels')
    val_img_dir = Path('dataset/val/images')
    val_label_dir = Path('dataset/val/labels')
    
    # Create validation directories
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all labeled images
    labeled_images = []
    for img_path in train_img_dir.glob('*.jpg'):
        label_path = train_label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            labeled_images.append(img_path)
    
    print(f"\nFound {len(labeled_images)} labeled images")
    
    # Randomly select 20% for validation
    num_val = int(len(labeled_images) * 0.2)
    val_images = random.sample(labeled_images, num_val)
    
    # Move validation images and their labels
    for img_path in val_images:
        label_path = train_label_dir / (img_path.stem + '.txt')
        
        # Move image
        shutil.move(str(img_path), str(val_img_dir / img_path.name))
        # Move label
        shutil.move(str(label_path), str(val_label_dir / label_path.name))
    
    print(f"Moved {num_val} images to validation set")

def train_model():
    """Train YOLOv8 model on our dataset."""
    print("\n=== Starting Model Training ===")
    print("Training will take 10-15 minutes. You'll see updates every few seconds.")
    print("\nProgress indicators:")
    print("- Epoch: Training cycle (will go from 1 to 10)")
    print("- box_loss: How well it's learning to find hands")
    print("- cls_loss: How well it's learning rock/paper/scissors")
    print("\nTraining is running if you see these numbers updating.")
    print("DO NOT close this window!\n")
    time.sleep(3)  # Give time to read instructions
    
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    try:
        print("Loading model and starting training...")
        time.sleep(1)
        
        results = model.train(
            data='dataset.yaml',          # Path to data config file
            epochs=10,                    # Number of epochs
            imgsz=640,                    # Image size
            patience=3,                   # Early stopping patience
            batch=8,                      # Batch size
            device='cpu',                 # Use CPU
            project='runs/train',         # Save results to runs/train
            name='rps_detector',          # Name of the experiment
            exist_ok=True,                # Overwrite existing experiment
            pretrained=True,              # Use pretrained weights
            verbose=True,                 # Print training progress
            plots=True                    # Generate performance plots
        )
        
        # Save the trained model
        print("\nTraining complete! Saving model...")
        model.export(format='onnx')
        print("\nSuccess! Model saved to runs/train/rps_detector/weights/")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nPlease try running the training again.")
        raise

def main():
    """Main function."""
    print("=== Rock Paper Scissors Detector Training ===")
    print("\nPreparing validation set...")
    prepare_validation_set()
    train_model()

if __name__ == "__main__":
    main() 