import cv2
import numpy as np
import pickle
import sys

sys.stdout.reconfigure(line_buffering=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.read('face_model.yml')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

with open('face_labels.pkl', 'rb') as f:
    label_to_name, _ = pickle.load(f)

print("Labels in model:", label_to_name)

# Test with the training images - simulating webcam quality
for img_name in ['straight.jpg', 'left.jpg', 'right.jpg']:
    img = cv2.imread('known_faces/CS-B_CS-B-009/' + img_name)
    if img is None:
        print(img_name + ": FAILED TO LOAD")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
    if len(faces) == 0:
        h, w = gray.shape
        # Try center crop as face region
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.1)
        faces = np.array([[margin_x, margin_y, w - 2*margin_x, h - 2*margin_y]])
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        roi = clahe.apply(roi)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        label, conf = recognizer.predict(roi)
        name = label_to_name.get(label, "Unknown")
        print(img_name + " (original): " + name + " conf=" + str(round(conf, 1)))
        
        # Simulate webcam quality: downscale then upscale
        small = cv2.resize(roi, (50, 50))
        webcam_sim = cv2.resize(small, (200, 200))
        webcam_sim = clahe.apply(webcam_sim)
        label2, conf2 = recognizer.predict(webcam_sim)
        name2 = label_to_name.get(label2, "Unknown")
        print(img_name + " (webcam-sim): " + name2 + " conf=" + str(round(conf2, 1)))

print("DONE")
