"""
Logging utility for saving detection results and images.
"""
import csv
import cv2
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DETECTIONS_DIR, LOG_FILE, LOG_HEADERS

class DetectionLogger:
    def __init__(self):
        """Initialize the detection logger."""
        self._init_log_file()
    
    def _init_log_file(self):
        """Create or verify the log file with headers."""
        if not LOG_FILE.exists():
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(LOG_HEADERS)
    
    def save_detection(self, frame, label, confidence):
        """
        Save a detection result and the corresponding frame.
        
        Args:
            frame: The captured frame (numpy array)
            label: The detected class label (str)
            confidence: Detection confidence score (float)
        
        Returns:
            str: Path to the saved image
        """
        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{timestamp}_{label}.jpg"
        image_path = DETECTIONS_DIR / image_filename
        
        # Save the frame
        cv2.imwrite(str(image_path), frame)
        
        # Log the detection
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                label,
                f"{confidence:.3f}",
                str(image_path)
            ])
        
        return str(image_path)

    def get_latest_detections(self, n=5):
        """
        Get the n most recent detections.
        
        Args:
            n (int): Number of detections to retrieve
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        try:
            with open(LOG_FILE, 'r') as f:
                reader = csv.DictReader(f)
                detections = list(reader)[-n:]
        except FileNotFoundError:
            pass
        return detections 