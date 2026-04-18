"""Quick debug script to check actual confidence scores and anti-spoofing results."""
import cv2
import os
import numpy as np
import pickle
import time
import anti_spoofing

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
recognizer.read("face_model.yml")
with open("face_labels.pkl", "rb") as f:
    label_to_name, name_to_label = pickle.load(f)
print("Model labels:", label_to_name)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Open camera with DirectShow backend (more reliable on Windows)
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not video.isOpened():
    print("Cannot open camera")
    exit()

print("Capturing frames - show your face to camera. Press 'q' to quit.")
time.sleep(1)

frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Check liveness
        live = anti_spoofing.check_liveness(roi_color)

        # Predict
        roi = cv2.resize(roi_gray, (200, 200))
        roi = clahe.apply(roi)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        label, confidence = recognizer.predict(roi)
        name = label_to_name.get(label, "???")

        failed = live.get("failed", [])
        is_live = live["is_live"]
        
        # Show on frame
        color = (0, 255, 0) if (confidence < 105 and is_live) else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        info = f"{name} conf={confidence:.0f} live={is_live}"
        cv2.putText(frame, info, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if frame_count % 10 == 0:
            print(f"face={name}, confidence={confidence:.1f}, is_live={is_live}, failed_checks={failed}")
    
    cv2.imshow("Debug Camera - press q to quit", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    frame_count += 1

video.release()
cv2.destroyAllWindows()
print("Done")
