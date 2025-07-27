"""
Play Rock Paper Scissors against the computer using webcam detection.
"""
import cv2
import time
import random
from pathlib import Path
from ultralytics import YOLO

# Load the trained model
model_path = Path('runs/train/rps_detector/weights/best.pt')
model = YOLO(model_path)

# Game rules
RULES = {
    'rock': {'rock': 'Tie!', 'paper': 'Computer wins!', 'scissors': 'You win!'},
    'paper': {'rock': 'You win!', 'paper': 'Tie!', 'scissors': 'Computer wins!'},
    'scissors': {'rock': 'Computer wins!', 'paper': 'You win!', 'scissors': 'Tie!'}
}

def get_computer_move():
    """Get random move from computer."""
    return random.choice(['rock', 'paper', 'scissors'])

def draw_text(frame, text, position, color=(255, 255, 255)):
    """Draw text with background on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw background rectangle
    padding = 5
    cv2.rectangle(
        frame,
        (position[0] - padding, position[1] - text_size[1] - padding),
        (position[0] + text_size[0] + padding, position[1] + padding),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        color,
        thickness
    )

def main():
    """Main game loop."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set up game state
    countdown = 0
    game_result = ""
    computer_move = None
    last_detection_time = time.time()
    detection_threshold = 0.1  # Lowered confidence threshold
    
    print("\nRock Paper Scissors Game!")
    print("Show your hand gesture to the camera")
    print("Press 'q' to quit")
    print("Press 'p' to play a round")
    print("\nTips:")
    print("- Hold your hand steady")
    print("- Make sure there's good lighting")
    print("- Keep your hand in frame")
    print("- Make clear gestures")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run detection
        results = model(frame, conf=detection_threshold, verbose=False)
        
        # Process detections
        current_gesture = None
        if len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                # Get highest confidence detection
                box = result.boxes[0]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                current_gesture = model.names[class_id]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{current_gesture}: {confidence:.2f}"
                draw_text(frame, label, (x1, y1 - 10))
                
                # Update last detection time
                last_detection_time = time.time()
        
        # Draw game information
        if countdown > 0:
            draw_text(frame, str(countdown), (frame.shape[1]//2 - 20, 50), (0, 255, 255))
        elif computer_move:
            draw_text(frame, f"Computer played: {computer_move}", (10, 30))
            if game_result:
                draw_text(frame, game_result, (10, 70), (0, 255, 0))
        
        # Show current detection
        if current_gesture:
            draw_text(frame, f"Detected: {current_gesture}", (10, frame.shape[0] - 50))
        
        # Show instructions
        draw_text(frame, "Press 'p' to play, 'q' to quit", (10, frame.shape[0] - 10))
        
        # Display frame
        cv2.imshow('Rock Paper Scissors', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Start new game round
            countdown = 3
            computer_move = get_computer_move()
            game_result = ""
            start_time = time.time()
            
            # Countdown
            while countdown > 0:
                if time.time() - start_time >= 1:
                    countdown -= 1
                    start_time = time.time()
                
                # Keep showing video during countdown
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    draw_text(frame, str(countdown), (frame.shape[1]//2 - 20, 50), (0, 255, 255))
                    cv2.imshow('Rock Paper Scissors', frame)
                    cv2.waitKey(1)
            
            # Get player's move (wait for detection)
            detection_start = time.time()
            player_move = None
            best_confidence = 0
            
            # Wait longer for move detection
            while time.time() - detection_start < 3:  # Increased wait time to 3 seconds
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    results = model(frame, conf=detection_threshold, verbose=False)
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        box = results[0].boxes[0]
                        confidence = float(box.conf[0])
                        if confidence > best_confidence:
                            best_confidence = confidence
                            class_id = int(box.cls[0])
                            player_move = model.names[class_id]
                        
                        # Draw current detection
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{model.names[class_id]}: {confidence:.2f}"
                        draw_text(frame, label, (x1, y1 - 10))
                    
                    draw_text(frame, "Show your move!", (frame.shape[1]//2 - 100, 50), (0, 255, 255))
                    cv2.imshow('Rock Paper Scissors', frame)
                    cv2.waitKey(1)
            
            # Determine winner
            if player_move:
                game_result = f"You played {player_move}. " + RULES[player_move][computer_move]
            else:
                game_result = "No move detected! Try again."
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 