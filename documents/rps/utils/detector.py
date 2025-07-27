"""
Detector utility for recognizing rock, paper, scissors hand gestures.
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import CONFIDENCE_THRESHOLD, DETECTION_CLASSES

class HandGestureDetector:
    def __init__(self):
        """Initialize the hand gesture detector."""
        self.model = YOLO('yolov8n.pt')
        # For now, we'll use the default YOLO model which can detect hands
        # Later we'll train it specifically for rock, paper, scissors
        
    def preprocess_frame(self, frame):
        """
        Preprocess the frame for detection.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # YOLO model handles preprocessing internally
        return frame
    
    def detect(self, frame):
        """
        Detect hand gestures in the frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            tuple: (label, confidence, bbox) or (None, 0, None) if no detection
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        # Process results
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the highest confidence detection
            boxes = results[0].boxes
            conf = float(boxes.conf[0])
            
            if conf > CONFIDENCE_THRESHOLD:
                # For now, we'll detect any hand as "hand"
                # Later we'll classify specific gestures
                label = "hand"
                bbox = boxes.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                return label, conf, bbox
        
        return None, 0, None
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            "name": "YOLOv8n Hand Detector",
            "classes": ["hand"],  # For now, just detecting hands
            "threshold": CONFIDENCE_THRESHOLD
        } 