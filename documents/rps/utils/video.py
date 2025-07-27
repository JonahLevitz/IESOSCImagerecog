yes"""
Video capture utility for handling webcam input and frame processing.
"""
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    WINDOW_NAME, COLORS, FONT_SCALE, FONT_THICKNESS,
    BOX_THICKNESS
)

class VideoCapture:
    def __init__(self):
        """Initialize video capture with the default webcam."""
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self._configure_capture()
        
    def _configure_capture(self):
        """Configure video capture parameters."""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
    
    def read_frame(self):
        """
        Read a frame from the video capture.
        
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the captured image
        """
        return self.cap.read()
    
    def draw_detection(self, frame, label, confidence, bbox):
        """
        Draw detection results on the frame.
        
        Args:
            frame: The frame to draw on
            label: The detected class label
            confidence: Detection confidence score
            bbox: Bounding box coordinates (x1, y1, x2, y2)
        
        Returns:
            numpy.ndarray: The annotated frame
        """
        x1, y1, x2, y2 = map(int, bbox)
        color = COLORS.get(label.lower(), COLORS['text'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # Draw label and confidence
        label_text = f"{label}: {confidence:.2f}"
        text_size = cv2.getTextSize(
            label_text, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            FONT_SCALE, 
            FONT_THICKNESS
        )[0]
        
        # Draw text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0], y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            COLORS['text'],
            FONT_THICKNESS
        )
        
        return frame
    
    def show_frame(self, frame):
        """
        Display the frame in a window.
        
        Args:
            frame: The frame to display
        """
        cv2.imshow(WINDOW_NAME, frame)
    
    def release(self):
        """Release the video capture and destroy all windows."""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release() 