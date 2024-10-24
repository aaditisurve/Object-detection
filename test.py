import cv2
import cvzone
import math
import time
import pyttsx3
import numpy as np
from ultralytics import YOLO

# Change the source to 0 for webcam
cap = cv2.VideoCapture(0) 
# Optional: Set resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

model = YOLO('../Yolo-Weights/yolo11l.pt')

engine = pyttsx3.init()

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for YOLO

# Function to apply gamma correction
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to estimate distance (same as before)
def estimate_distance(x1, y1, x2, y2, known_object_height=1.7):
    object_pixel_height = y2 - y1
    focal_length = 700  # Focal length example
    if object_pixel_height > 0:
        distance = (known_object_height * focal_length) / object_pixel_height
        return round(distance, 2)
    return None

# Function to check if an object is too close
def is_too_close(distance, threshold=2):
    return distance and distance < threshold

# Function to trigger voice alert
def trigger_audio_alert(message):
    engine.say(message)
    engine.runAndWait()

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    # Apply night vision processing (CLAHE and Gamma Correction)
    img_clahe = apply_clahe(img)
    img_gamma_corrected = adjust_gamma(img_clahe, gamma=1.5)

    # Run YOLO detection on the preprocessed image
    results = model(img_gamma_corrected, stream=True)
    
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(box.cls[0])
            
            # Estimate distance to the detected object
            distance = estimate_distance(x1, y1, x2, y2)

            # Check if the object is too close, regardless of the class
            if conf > 0.5:  # You can adjust the confidence threshold as needed
                if is_too_close(distance):
                    trigger_audio_alert(f'{classNames[cls]} is too close! Distance: {distance} meters')

            # Display class name, confidence, and distance on the frame
            cvzone.putTextRect(img, f'{classNames[cls]} {conf} Dist: {distance}m',
                               (max(0, x1), max(35, y1)), scale=1, thickness=2)

    # Show the output
    cv2.imshow("Image", img)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
