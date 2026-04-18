# debug_recognition.py
"""
Test recognition accuracy on your test images before running live.
Put a few test photos in a test_images/ folder and run this script.
"""
import cv2
import face_recognition
import pickle
import numpy as np
import os

# Load model
with open("face_encodings.pkl", "rb") as f:
    known_encodings = pickle.load(f)

print(f"Model loaded: {len(known_encodings)} people enrolled")
print(f"Encodings per person:")
for name, encs in known_encodings.items():
    print(f"  {name}: {len(encs)} encodings")

# Test on a test image if provided
test_dir = "test_images"
if os.path.exists(test_dir):
    THRESHOLD = 0.55
    for img_file in os.listdir(test_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(test_dir, img_file)
        image = face_recognition.load_image_file(img_path)
        locations = face_recognition.face_locations(image)
        
        if not locations:
            print(f"\n{img_file}: No face detected")
            continue
        
        encodings = face_recognition.face_encodings(image, locations)
        
        for enc in encodings:
            results = {}
            for person_name, stored_encs in known_encodings.items():
                distances = face_recognition.face_distance(stored_encs, enc)
                results[person_name] = float(np.min(distances))
            
            # Sort by distance
            sorted_results = sorted(results.items(), key=lambda x: x[1])
            
            best_name, best_dist = sorted_results[0]
            match = best_name if best_dist <= THRESHOLD else "Unknown"
            
            print(f"\n{img_file}:")
            print(f"  Best match: {match} (distance: {best_dist:.3f})")
            print(f"  All distances:")
            for name, dist in sorted_results:
                status = "MATCH" if dist <= THRESHOLD else "X"
                print(f"    {status} {name}: {dist:.3f}")
else:
    print("\nCreate a 'test_images/' folder with some test photos to verify accuracy.")
    print("Example: put a photo of each student named 'student1_test.jpg'")
