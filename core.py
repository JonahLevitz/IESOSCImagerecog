"""
Emergency Detection System for Drone-based Medical Supply Delivery
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class EmergencyDetector:
    """Detector for emergency situations like fires, crashes, and medical emergencies."""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize the emergency detector.
        
        Args:
            model_path (str): Path to YOLO model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Classes we're interested in from COCO dataset
        self.emergency_classes = {
            'fire': ['fire'],  # Custom class if using trained model
            'crash': ['car', 'truck'],  # Look for multiple vehicles in close proximity
            'medical': ['person']  # Will need custom training for "person lying down"
        }
        
        # Setup detection directory
        self.detection_dir = Path('detections')
        self.detection_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging directory and file."""
        self.log_file = self.detection_dir / 'emergency_log.csv'
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write('timestamp,type,confidence,image_path,details\n')
    
    def _log_detection(self, emergency_type, confidence, image_path, details=''):
        """Log a detection with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f'{timestamp},{emergency_type},{confidence:.3f},{image_path},{details}\n')
    
    def _save_detection_image(self, frame, emergency_type):
        """Save detection frame with timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = self.detection_dir / f'{emergency_type}_{timestamp}.jpg'
        cv2.imwrite(str(image_path), frame)
        return image_path
    
    def process_frame(self, frame):
        """
        Process a single frame for emergency detection.
        
        Args:
            frame (np.ndarray): Input frame from camera/video
            
        Returns:
            tuple: (processed_frame, detections)
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        detections = []
        processed_frame = frame.copy()
        
        if len(results) > 0:
            # Process each detection
            boxes = results[0].boxes
            for box in boxes:
                # Get detection info
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                    
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                bbox = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                
                # Check if detection is relevant for emergency detection
                emergency_type = None
                for e_type, classes in self.emergency_classes.items():
                    if class_name in classes:
                        emergency_type = e_type
                        break
                
                if emergency_type:
                    # Save detection
                    image_path = self._save_detection_image(frame, emergency_type)
                    
                    # Log detection
                    details = f'class={class_name}'
                    self._log_detection(emergency_type, conf, image_path, details)
                    
                    # Draw detection
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 0, 255)  # Red for emergency
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f'{emergency_type}: {conf:.2f}'
                    cv2.putText(processed_frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detections.append({
                        'type': emergency_type,
                        'confidence': conf,
                        'bbox': bbox,
                        'class_name': class_name
                    })
        
        return processed_frame, detections
