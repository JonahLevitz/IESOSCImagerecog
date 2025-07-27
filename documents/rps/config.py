"""
Configuration settings for the Rock Paper Scissors detection system.
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent

# Directories
MODELS_DIR = ROOT_DIR / "models"
DETECTIONS_DIR = ROOT_DIR / "detections"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for dir_path in [MODELS_DIR, DETECTIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
DETECTION_CLASSES = ["rock", "paper", "scissors"]

# Video settings
CAMERA_ID = 0  # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Logging settings
LOG_FILE = LOGS_DIR / "detections.csv"
LOG_HEADERS = ["timestamp", "label", "confidence", "image_path"]

# Display settings
WINDOW_NAME = "Rock Paper Scissors Detection"
FONT = "FONT_HERSHEY_SIMPLEX"
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# Colors (BGR format)
COLORS = {
    "rock": (255, 0, 0),      # Blue
    "paper": (0, 255, 0),     # Green
    "scissors": (0, 0, 255),  # Red
    "text": (255, 255, 255)   # White
} 