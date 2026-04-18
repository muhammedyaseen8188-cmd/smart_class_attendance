# functions.py — COMPLETE REPLACEMENT
"""
Face recognizer using face_recognition library (dlib deep learning backend).
99.38% accuracy on LFW benchmark vs LBPH's ~60-70% on 3 images.

Drop-in replacement: same class name, same methods, same return values.
"""
import cv2
import os
import time
import numpy as np
import pickle
import face_recognition
from pathlib import Path


class FaceRecognizer:
    """
    Face recognizer using 128-dimensional deep face embeddings.
    
    HOW IT WORKS (vs old LBPH):
    
    OLD (LBPH): Compare pixel brightness patterns in 8x8 grid squares.
    Fails with different lighting, angles, expressions. Needs 20+ images.
    
    NEW (this): A deep CNN converts any face into a 128-number vector.
    The numbers encode the geometric relationships between facial features
    (distance between eyes, jaw width, nose shape, etc.). Two photos of
    the same person have vectors very close together. Different people
    have vectors far apart — regardless of lighting or angle.
    
    Recognition = find the stored vector closest to the live vector.
    Threshold 0.55 = vectors must be within 0.55 Euclidean distance.
    Works reliably with just 3-5 images if they're good quality.
    """

    def __init__(self, known_faces_dir="known_faces",
                 model_path="face_encodings.pkl"):
        self.known_faces_dir = known_faces_dir
        self.model_path = model_path

        # Face detector (still using OpenCV Haar for speed in video frames)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Known face data
        self.known_encodings = {}   # person_name → [list of 128-dim vectors]
        self.is_trained = False

        # CLAHE for lighting normalisation (still useful for display)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Consistency tracking: same person must appear N consecutive frames
        # This prevents a single-frame false positive from marking attendance
        self.recognition_history = {}   # position_key → [recent names]
        self.confirmed_faces = {}       # position_key → confirmed name
        self.CONFIRM_FRAMES = 8         # Increased from 5 for reliability

        # Distance threshold: how close must embeddings be to match?
        # 0.55 = strict (fewer false positives, may miss some angles)
        # 0.60 = balanced (recommended default)
        # 0.65 = permissive (more matches, higher false positive risk)
        self.DISTANCE_THRESHOLD = 0.55

    def load_model(self):
        """Load pre-computed face encodings from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.known_encodings = pickle.load(f)
                self.is_trained = len(self.known_encodings) > 0
                if self.is_trained:
                    total = sum(len(v) for v in self.known_encodings.values())
                    print(f"Loaded {len(self.known_encodings)} people "
                          f"({total} encodings) from {self.model_path}")
                return self.is_trained
            except Exception as e:
                print(f"Error loading model: {e}")
        return False

    def load_known_faces(self):
        """
        Full pipeline: augment images → compute encodings → save model.
        Called when no pre-trained model exists.
        """
        print("Running augmentation and training pipeline...")
        
        # Step 1: Augment the 3 images per person into 30+
        import augment_faces
        augment_faces.augment_known_faces_directory(
            known_faces_dir=self.known_faces_dir,
            augmented_dir="known_faces_augmented"
        )

        # Step 2: Train from augmented directory
        import train_model
        success = train_model.train_from_directory(
            faces_dir="known_faces_augmented",
            output_path=self.model_path
        )

        if success:
            return self.load_model()
        return False

    def _get_best_match(self, live_encoding):
        """
        Find the best matching person for a live face encoding.
        
        Returns: (person_name, distance) or ("Unknown", 1.0)
        
        Method: Compare live encoding against ALL stored encodings
        for every enrolled person. For each person, take the MINIMUM
        distance (best match) across all their stored encodings.
        Then pick the person with the overall minimum distance.
        
        This is more robust than comparing against a single stored
        encoding — it handles lighting and angle variations by having
        many reference points.
        """
        if not self.known_encodings:
            return "Unknown", 1.0

        best_name = "Unknown"
        best_distance = self.DISTANCE_THRESHOLD  # Only accept within threshold

        for person_name, stored_encodings in self.known_encodings.items():
            if not stored_encodings:
                continue

            # Compare live encoding against all stored encodings for this person
            distances = face_recognition.face_distance(
                stored_encodings, live_encoding
            )

            # Take the minimum distance (best match for this person)
            min_distance = float(np.min(distances))

            if min_distance < best_distance:
                best_distance = min_distance
                best_name = person_name

        return best_name, best_distance

    def recognize_faces(self, frame, confidence_threshold=None):
        """
        Recognize faces in a video frame.
        
        Maintains backward compatibility: same signature as old LBPH version.
        confidence_threshold parameter is ignored (kept for compatibility).
        
        Returns:
            frame: annotated image
            recognized_names: list of CONFIRMED person names
        """
        recognized_names = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face locations using OpenCV (fast)
        detected_faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        active_keys = set()

        for (x, y, w, h) in detected_faces:
            # Position key for consistency tracking (50px grid quantisation)
            pos_key = (x // 50, y // 50)
            active_keys.add(pos_key)

            # Extract face ROI for liveness and display
            face_roi_color = frame[y:y+h, x:x+w]

            # ── ANTI-SPOOFING ─────────────────────────────────────────────
            try:
                import anti_spoofing
                liveness = anti_spoofing.check_liveness(face_roi_color)
                if not liveness.get('is_live', True):
                    label_text = f"SPOOF ({liveness.get('spoof_type', '?')})"
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), color, cv2.FILLED)
                    cv2.putText(frame, label_text, (x+6, y+h+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    continue
            except ImportError:
                pass  # anti_spoofing module not available

            # ── FACE RECOGNITION ──────────────────────────────────────────
            if self.is_trained:
                # Convert face region for face_recognition
                # face_recognition expects (top, right, bottom, left) format
                face_location = [(y, x+w, y+h, x)]

                encodings = face_recognition.face_encodings(
                    rgb_frame, face_location, num_jitters=1
                )

                if not encodings:
                    label_text = "No encoding"
                    color = (128, 128, 128)
                else:
                    live_encoding = encodings[0]
                    best_name, distance = self._get_best_match(live_encoding)

                    # ── CONSISTENCY CHECK ──────────────────────────────────
                    # Must be same person for CONFIRM_FRAMES in a row
                    if best_name != "Unknown":
                        if pos_key not in self.recognition_history:
                            self.recognition_history[pos_key] = []
                        self.recognition_history[pos_key].append(best_name)
                        self.recognition_history[pos_key] = \
                            self.recognition_history[pos_key][-self.CONFIRM_FRAMES:]

                        history = self.recognition_history[pos_key]

                        if pos_key in self.confirmed_faces:
                            # Already confirmed — use cached name
                            name = self.confirmed_faces[pos_key]
                            recognized_names.append(name)
                            color = (0, 255, 0)
                            pct = int((1 - distance) * 100)
                            label_text = f"{name} ({pct}%)"

                        elif (len(history) >= self.CONFIRM_FRAMES and
                              len(set(history)) == 1):
                            # Confirmed: same person for N consecutive frames
                            name = best_name
                            self.confirmed_faces[pos_key] = name
                            recognized_names.append(name)
                            color = (0, 255, 0)
                            pct = int((1 - distance) * 100)
                            label_text = f"{name} ({pct}%)"

                        else:
                            # Still verifying
                            color = (0, 255, 255)
                            progress = len(history)
                            label_text = f"Verifying... {progress}/{self.CONFIRM_FRAMES}"
                    else:
                        # Unknown face
                        color = (0, 0, 255)
                        label_text = "Unknown"
                        # Clear history for this position
                        self.recognition_history.pop(pos_key, None)
                        self.confirmed_faces.pop(pos_key, None)
            else:
                label_text = "No model loaded"
                color = (128, 128, 128)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), color, cv2.FILLED)
            cv2.putText(frame, label_text, (x+6, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Clean up stale position tracking
        stale = [k for k in list(self.recognition_history.keys())
                 if k not in active_keys]
        for k in stale:
            self.recognition_history.pop(k, None)
            self.confirmed_faces.pop(k, None)

        return frame, recognized_names

    def capture_face(self, frame, person_name):
        """Capture and save a face. Identical interface to old version."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(detected_faces) == 0:
            print("No face detected in frame!")
            return False

        if len(detected_faces) > 1:
            print("Multiple faces detected! Only one person should be in frame.")
            return False

        person_dir = os.path.join(self.known_faces_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name}_{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)

        cv2.imwrite(filepath, frame)
        print(f"Saved to {filepath}")
        return True

    def preprocess_face(self, gray_face):
        """Legacy compatibility — not used in new pipeline."""
        return self.clahe.apply(gray_face)