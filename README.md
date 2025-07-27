# IESOSC Image Recognition System

Emergency Detection System for Drone-based Medical Supply Delivery

## Overview

This project implements an AI-powered image recognition system designed to detect emergency situations such as fires, crashes, and medical emergencies. The system is intended for use in drone-based medical supply delivery scenarios.

## Features

- **Real-time Emergency Detection**: Detects fires, crashes, and medical emergencies
- **Webcam Integration**: Works with live camera feeds
- **Video File Support**: Can process pre-recorded video files
- **Detection Logging**: Automatically logs all detections with timestamps
- **Image Capture**: Saves detection frames for later analysis
- **Configurable Confidence**: Adjustable detection confidence thresholds

## System Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Webcam or video input source

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IESOSC_Imagerecognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the emergency detection system with default settings:

```bash
python run_detection.py
```

### Advanced Usage

Run with custom parameters:

```bash
python run_detection.py --source 0 --model yolov8n.pt --conf 0.5
```

Parameters:
- `--source`: Video source (0 for webcam, or path to video file)
- `--model`: Path to YOLO model weights
- `--conf`: Confidence threshold (0.0 to 1.0)

### Examples

```bash
# Use webcam with custom confidence
python run_detection.py --conf 0.7

# Process video file
python run_detection.py --source path/to/video.mp4

# Use custom model
python run_detection.py --model path/to/custom_model.pt
```

## Project Structure

```
IESOSC_Imagerecognition/
├── core.py                 # Main detection engine
├── run_detection.py        # Command-line interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── documents/             # Additional documentation
│   └── rps/              # Rock Paper Scissors detection (separate project)
└── firedetection/         # Fire detection specific modules
```

## Detection Classes

The system is configured to detect the following emergency situations:

- **Fire**: Detects fire-related objects and situations
- **Crash**: Detects vehicle crashes and accidents
- **Medical**: Detects people in potential medical emergency situations

## Output

The system provides:
- Real-time video feed with detection overlays
- Detection logs saved to `detections/emergency_log.csv`
- Captured images of detections saved to `detections/` directory
- Console output with detection details

## Configuration

You can modify the detection parameters in `core.py`:

- `conf_threshold`: Minimum confidence for detections
- `emergency_classes`: Classes to monitor for each emergency type
- Detection directory and logging settings

## Development

### Adding New Detection Classes

1. Modify the `emergency_classes` dictionary in `EmergencyDetector.__init__()`
2. Add new emergency types and their corresponding COCO classes
3. Update the detection logic as needed

### Training Custom Models

1. Prepare your dataset in YOLO format
2. Train using Ultralytics YOLO
3. Update the model path in the detector initialization

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support information here] 