"""
Real-Time Sign Language Recognition
Uses a trained CNN model to recognize ASL letters from webcam feed.
Press 'q' to quit
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = 'sign_language_model.h5'  # or 'sign_language_model.keras'
IMAGE_SIZE = 28
CONFIDENCE_THRESHOLD = 0.5
DISPLAY_HISTORY = 15  # Number of frames to average predictions

# Label mapping A-Z
LABEL_TO_LETTER = {i: chr(65 + i) for i in range(26)}

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"✗ Error: Model file '{MODEL_PATH}' not found!")
    print("  Please train the model first using sign_language_training.ipynb")
    exit(1)

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================
def preprocess_frame(frame, target_size=(28, 28)):
    """
    Preprocess a frame for the sign language model.
    
    Args:
        frame: BGR image from OpenCV
        target_size: target size (28x28)
    
    Returns:
        Preprocessed image array ready for model input
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized = cv2.resize(gray, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model input (1, 28, 28, 1)
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    return reshaped

# ============================================================================
# MAIN RECOGNITION LOOP
# ============================================================================
def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open camera device")
        exit(1)
    
    print("✓ Camera opened successfully")
    print("Press 'q' to quit\n")
    
    # History for smoothing predictions
    prediction_history = deque(maxlen=DISPLAY_HISTORY)
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # FPS counter
    prev_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Define ROI (region of interest) - center area for hand
            h, w = frame.shape[:2]
            roi_size = 200
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Preprocess ROI
            processed = preprocess_frame(roi)
            
            # Make prediction
            prediction = model.predict(processed, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Store prediction history
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(predicted_class)
            
            # Get most common prediction from history (smoothing)
            if prediction_history:
                from collections import Counter
                smoothed_class = Counter(prediction_history).most_common(1)[0][0]
                predicted_letter = LABEL_TO_LETTER[smoothed_class]
            else:
                predicted_letter = "?"
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), 
                         (0, 255, 0), 2)
            
            # Prepare display text
            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = f"{predicted_letter} ({confidence:.2f})"
                text_color = (0, 255, 0)  # Green
            else:
                display_text = "No hand detected"
                text_color = (0, 165, 255)  # Orange
            
            # Draw predictions on frame
            cv2.putText(frame, display_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_width = 300
            bar_height = 20
            bar_x, bar_y = 10, 120
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                filled_width = int(bar_width * confidence)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                             (0, 255, 0), -1)
            
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n✓ Quitting...")
                break
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released and windows closed")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Sign Language Recognition - Real-Time")
    print("=" * 60)
    main()
    print("\nDone!")
