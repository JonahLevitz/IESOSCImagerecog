"""
Script to download and setup YOLOv8 model for hand gesture detection.
"""
from pathlib import Path
import sys
from ultralytics import YOLO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR

def download_model():
    """Download YOLOv8n model."""
    print("Downloading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("Model downloaded successfully!")
    return model

if __name__ == "__main__":
    model = download_model()
    print("Model setup complete!") 