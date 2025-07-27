# Rock Paper Scissors Detection System

This project uses computer vision to detect and classify Rock, Paper, and Scissors hand gestures in real-time using a webcam. It serves as a proof of concept for a larger drone-based emergency detection system.

## Features

- Real-time hand gesture detection using webcam
- Classification of Rock, Paper, and Scissors gestures
- Detection logging with timestamps
- Snapshot saving of detected gestures
- Configurable confidence threshold
- Live video feed with annotated detections

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- NumPy

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment (if not already activated)
2. Run the main script:
```bash
python main.py
```

3. Hold your hand in front of the webcam making one of these gestures:
   - Closed fist for "Rock"
   - Open palm for "Paper"
   - Two fingers for "Scissors"

4. Press 'q' to quit the program

## Project Structure

```
.
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── models/                # Model files and weights
├── utils/                 # Utility functions
│   ├── detector.py       # Detection and inference logic
│   ├── video.py         # Video capture and processing
│   └── logger.py        # Logging and file saving
├── detections/           # Saved detection images
└── logs/                 # Detection log files
```

## Future Extensions

This project will be extended to support drone-based emergency situation detection, including:
- Fire detection
- Car crash detection
- Medical emergency detection 