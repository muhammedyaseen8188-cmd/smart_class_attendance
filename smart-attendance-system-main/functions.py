import time
import cv2
import os
import numpy as np
import pickle
import anti_spoofing

def draw(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    parts = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in parts:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

def detect_face(img, faceCascade, eyeCascade):
    coords = draw(img, faceCascade, 1.1, 4, (255, 0, 0), "Face")
    
    if len(coords) == 4:
        roi_image = img[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[2]]
        coords = draw(roi_image, eyeCascade, 1.1, 8, (0, 0, 255), "Eyes")

    return img


class FaceRecognizer:
    """Face recognizer using OpenCV's LBPH algorithm - pip installable only"""
    
    def __init__(self, known_faces_dir="known_faces", model_path="face_model.yml"):
        self.known_faces_dir = known_faces_dir
        self.model_path = model_path
        self.labels_path = "face_labels.pkl"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        # LBPH parameters - radius=2 captures more texture for better discrimination
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        self.label_to_name = {}
        self.name_to_label = {}
        self.reference_faces = {}  # label -> list of preprocessed reference face images
        self.is_trained = False
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Consistency tracking: require same person N times in a row
        self.recognition_history = {}  # face_position_key -> [list of recent names]
        self.confirmed_faces = {}  # face_position_key -> confirmed name
        self.CONFIRM_FRAMES = 5  # Must be same person for 5 frames
    
    def preprocess_face(self, gray_face):
        """Apply preprocessing to normalize lighting variations."""
        # Apply CLAHE for adaptive histogram equalization
        normalized = self.clahe.apply(gray_face)
        # Optional: Gaussian blur to reduce noise
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
        return normalized

    def augment_face(self, face_img):
        """Generate augmented versions including webcam-quality simulation."""
        augmented = [face_img]  # original
        h, w = face_img.shape[:2]

        # Horizontal flip
        augmented.append(cv2.flip(face_img, 1))

        # Brightness variations (simulate different lighting)
        for alpha in [0.6, 0.75, 0.85, 1.15, 1.3, 1.5]:
            adjusted = cv2.convertScaleAbs(face_img, alpha=alpha, beta=0)
            augmented.append(adjusted)

        # Small rotations
        for angle in [-12, -7, -3, 3, 7, 12]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            augmented.append(rotated)

        # Slight crops (zoom in)
        margin = int(0.06 * w)
        crops = [
            face_img[margin:, margin:],
            face_img[margin:, :w-margin],
            face_img[:h-margin, margin:],
            face_img[:h-margin, :w-margin],
        ]
        for crop in crops:
            augmented.append(cv2.resize(crop, (w, h)))

        # Simulate webcam quality: downscale then upscale (creates blur/artifacts)
        for scale in [0.25, 0.35, 0.5]:
            small = cv2.resize(face_img, (int(w * scale), int(h * scale)))
            back_up = cv2.resize(small, (w, h))
            augmented.append(back_up)
            # Also with brightness variation
            augmented.append(cv2.convertScaleAbs(back_up, alpha=0.8, beta=0))
            augmented.append(cv2.convertScaleAbs(back_up, alpha=1.2, beta=0))

        # Gaussian noise (simulate webcam noise)
        for sigma in [10, 20, 30]:
            noise = np.random.normal(0, sigma, face_img.shape).astype(np.int16)
            noisy = np.clip(face_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            augmented.append(noisy)

        # Flip + brightness combos
        flipped = cv2.flip(face_img, 1)
        for alpha in [0.75, 1.25]:
            augmented.append(cv2.convertScaleAbs(flipped, alpha=alpha, beta=0))
        # Flip + low-res
        small_flip = cv2.resize(flipped, (int(w * 0.35), int(h * 0.35)))
        augmented.append(cv2.resize(small_flip, (w, h)))

        return augmented

    def verify_face_all_persons(self, face_roi, lbph_label, lbph_confidence):
        """Cross-verify face against ALL persons' references.
        Uses a tiered approach based on LBPH confidence strength."""
        if not self.reference_faces:
            return True, lbph_label, 1.0, {}
        
        scores_by_label = {}
        for label, refs in self.reference_faces.items():
            best_score = -1.0
            for ref_face in refs:
                result = cv2.matchTemplate(face_roi, ref_face, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]
                best_score = max(best_score, score)
            scores_by_label[label] = best_score
        
        if not scores_by_label:
            return True, lbph_label, 1.0, {}
        
        sorted_scores = sorted(scores_by_label.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = sorted_scores[0]
        lbph_score = scores_by_label.get(lbph_label, -1.0)
        
        # Tier 1: Strong LBPH match (< 75) - trust LBPH, just need minimal template
        if lbph_confidence < 75:
            return lbph_score >= 0.05, lbph_label, lbph_score, scores_by_label
        
        # Tier 2: Good LBPH match (75-90) - need template agreement or close score
        if lbph_confidence < 90:
            if best_label == lbph_label and lbph_score >= 0.10:
                return True, lbph_label, lbph_score, scores_by_label
            # Accept if LBPH person's template is close to best (within 0.05)
            if lbph_score >= 0.10 and (best_score - lbph_score) <= 0.05:
                return True, lbph_label, lbph_score, scores_by_label
            return False, best_label, lbph_score, scores_by_label
        
        # Tier 3: Weak LBPH match (90+) - require strong template confirmation
        if best_label == lbph_label and lbph_score >= 0.20:
            return True, lbph_label, lbph_score, scores_by_label
        
        return False, best_label, lbph_score, scores_by_label

    def load_known_faces(self):
        """
        Load and train the recognizer with faces from the known_faces directory.
        
        Directory structure:
        known_faces/
            Person1/
                image1.jpg
                image2.jpg
            Person2/
                image1.jpg
        """
        faces = []
        labels = []
        original_faces = []  # non-augmented faces for histogram verification
        original_labels = []
        current_label = 0
        self.label_to_name = {}
        self.name_to_label = {}
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created '{self.known_faces_dir}' directory. Please add subfolders with person names containing their images.")
            return False
        
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            # Assign label to this person
            if person_name not in self.name_to_label:
                self.name_to_label[person_name] = current_label
                self.label_to_name[current_label] = person_name
                current_label += 1
            
            label = self.name_to_label[person_name]
            
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                
                # Check if it's an image file
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                try:
                    # Load image and convert to grayscale
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                        
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces - try frontal first, then profile cascade
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    # Try relaxed frontal params
                    if len(detected_faces) == 0:
                        detected_faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                        )
                    
                    # Try profile face cascade (for left/right looking photos)
                    if len(detected_faces) == 0:
                        detected_faces = self.profile_cascade.detectMultiScale(
                            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                        )
                    
                    # Try flipped image with profile cascade (for other direction)
                    if len(detected_faces) == 0:
                        flipped_gray = cv2.flip(gray, 1)
                        detected_faces = self.profile_cascade.detectMultiScale(
                            flipped_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
                        )
                        # Adjust x coordinates back for original image
                        if len(detected_faces) > 0:
                            iw = gray.shape[1]
                            for i in range(len(detected_faces)):
                                detected_faces[i][0] = iw - detected_faces[i][0] - detected_faces[i][2]
                    
                    if len(detected_faces) == 0:
                        print(f"  WARNING: No face detected in {image_name}, skipping")
                        continue
                    
                    # Use ONLY the largest detected face (most likely the actual subject)
                    largest = max(detected_faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest
                    
                    # Extract and resize face region
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    # Apply preprocessing for lighting normalization
                    face_roi = self.preprocess_face(face_roi)
                    # Store original for histogram verification
                    original_faces.append(face_roi.copy())
                    original_labels.append(label)
                    # Generate augmented versions for better training
                    augmented_faces = self.augment_face(face_roi)
                    for aug_face in augmented_faces:
                        faces.append(aug_face)
                        labels.append(label)
                    print(f"Loaded face for {person_name} from {image_name} ({len(augmented_faces)} samples)")
                        
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        
        if len(faces) == 0:
            print("No faces found to train! Add images to known_faces/<PersonName>/ folders.")
            return False
        
        # Train the recognizer
        print(f"Training recognizer with {len(faces)} face(s) for {len(self.label_to_name)} person(s)...")
        self.recognizer.train(faces, np.array(labels))
        self.is_trained = True
        
        # Store original (non-augmented) reference faces for verification
        # Also store webcam-simulated versions to match webcam quality
        self.reference_faces = {}
        for face, lbl in zip(original_faces, original_labels):
            if lbl not in self.reference_faces:
                self.reference_faces[lbl] = []
            self.reference_faces[lbl].append(face)
            # Add webcam-quality simulated versions (downscale then upscale)
            h, w = face.shape[:2]
            for scale in [0.25, 0.35, 0.5]:
                small = cv2.resize(face, (int(w * scale), int(h * scale)))
                back_up = cv2.resize(small, (w, h))
                back_up = self.preprocess_face(back_up)
                self.reference_faces[lbl].append(back_up)
            # Add brightness variations
            for alpha in [0.7, 0.85, 1.15, 1.3]:
                adjusted = cv2.convertScaleAbs(face, alpha=alpha, beta=0)
                self.reference_faces[lbl].append(adjusted)
        ref_count = sum(len(v) for v in self.reference_faces.values())
        print(f"  Stored {ref_count} reference face(s) for template verification")
        
        # Save the model and labels
        self.recognizer.write(self.model_path)
        with open(self.labels_path, 'wb') as f:
            pickle.dump((self.label_to_name, self.name_to_label, self.reference_faces), f)
        
        print("Training complete!")
        return True
    
    def load_model(self):
        """Load a previously trained model."""
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            try:
                self.recognizer.read(self.model_path)
                with open(self.labels_path, 'rb') as f:
                    data = pickle.load(f)
                    self.label_to_name = data[0]
                    self.name_to_label = data[1]
                    if len(data) >= 3 and isinstance(data[2], dict):
                        # Check if it's reference_faces or confidence_baselines
                        first_val = next(iter(data[2].values()), None)
                        if isinstance(first_val, list):
                            self.reference_faces = data[2]
                        else:
                            self.reference_faces = {}
                    else:
                        self.reference_faces = {}
                self.is_trained = True
                ref_count = sum(len(v) for v in self.reference_faces.values())
                print(f"Loaded model with {len(self.label_to_name)} person(s), {ref_count} reference face(s)")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False
    
    def recognize_faces(self, frame, confidence_threshold=110):
        """
        Recognize faces in a frame with consistency checking.
        A face must be recognized as the same person for multiple consecutive 
        frames before being confirmed - prevents false matches.
        
        Args:
            frame: BGR image from OpenCV (video frame)
            confidence_threshold: LBPH confidence cutoff (lower = stricter)
        
        Returns:
            frame: Image with rectangles and names drawn on recognized faces
            recognized_names: List of CONFIRMED names recognized in this frame
        """
        recognized_names = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Track which position keys are active this frame
        active_keys = set()
        
        for (x, y, w, h) in detected_faces:
            name = "Unknown"
            label_text = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            
            # Create a rough position key (quantized to 50px grid) for tracking
            pos_key = (x // 50, y // 50)
            active_keys.add(pos_key)
            
            # Extract face regions
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            # Check Liveness FIRST
            liveness = anti_spoofing.check_liveness(face_roi_color)
            
            if not liveness['is_live']:
                name = "FAKE / SPOOF"
                label_text = f"SPOOF ({liveness['spoof_type']})"
                color = (0, 0, 255)
            elif self.is_trained:
                face_roi = cv2.resize(face_roi_gray, (200, 200))
                face_roi = self.preprocess_face(face_roi)
                
                label, confidence = self.recognizer.predict(face_roi)
                best_name = self.label_to_name.get(label, "Unknown")
                
                # Step 1: LBPH confidence check (lower = better match)
                if confidence < confidence_threshold:
                    # Step 2: Cross-verify against ALL persons' references
                    verified, tmpl_label, tmpl_score, all_scores = self.verify_face_all_persons(face_roi, label, confidence)
                    tmpl_name = self.label_to_name.get(tmpl_label, "?") if tmpl_label is not None else "?"
                    
                    print(f"  [DEBUG] lbph={best_name}({confidence:.1f}) tmpl_best={tmpl_name} lbph_tmpl={tmpl_score:.3f} verified={verified}", flush=True)
                    
                    if verified:
                        # Add to history for this position
                        if pos_key not in self.recognition_history:
                            self.recognition_history[pos_key] = []
                        self.recognition_history[pos_key].append(best_name)
                        # Keep only last N entries
                        self.recognition_history[pos_key] = self.recognition_history[pos_key][-self.CONFIRM_FRAMES:]
                        
                        history = self.recognition_history[pos_key]
                        
                        # Check if already confirmed
                        if pos_key in self.confirmed_faces:
                            name = self.confirmed_faces[pos_key]
                            recognized_names.append(name)
                            color = (0, 255, 0)  # Green for confirmed
                            label_text = f"{name} ({confidence:.0f})"
                        elif len(history) >= self.CONFIRM_FRAMES and len(set(history)) == 1:
                            # Same person for N consecutive frames - CONFIRMED
                            name = best_name
                            self.confirmed_faces[pos_key] = name
                            recognized_names.append(name)
                            color = (0, 255, 0)  # Green for confirmed
                            label_text = f"{name} ({confidence:.0f})"
                        else:
                            # Not yet confirmed - show as verifying (yellow)
                            color = (0, 255, 255)  # Yellow for verifying
                            progress = len(history)
                            label_text = f"Verifying... {progress}/{self.CONFIRM_FRAMES}"
                    else:
                        label_text = f"Unknown ({confidence:.0f})"
                else:
                    print(f"  [DEBUG] REJECTED: lbph={best_name}({confidence:.1f}) > threshold({confidence_threshold})", flush=True)
                    label_text = f"Unknown ({confidence:.0f})"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label with name (background matches box color, black text)
            cv2.rectangle(frame, (x, y+h), (x+w, y+h+35), color, cv2.FILLED)
            cv2.putText(frame, label_text, (x+6, y+h+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Clean up stale tracking for faces that left the frame
        stale = [k for k in self.recognition_history if k not in active_keys]
        for k in stale:
            del self.recognition_history[k]
            self.confirmed_faces.pop(k, None)
        
        return frame, recognized_names
    
    def capture_face(self, frame, person_name):
        """
        Capture and save a face from the current frame.
        
        Args:
            frame: Current video frame
            person_name: Name of the person to save
        
        Returns:
            success: Boolean indicating if face was captured successfully
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(detected_faces) == 0:
            print("No face detected in frame!")
            return False
        
        if len(detected_faces) > 1:
            print("Multiple faces detected! Please ensure only one face is in frame.")
            return False
        
        # Create directory for the person if it doesn't exist
        person_dir = os.path.join(self.known_faces_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save the image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name}_{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Saved face image to {filepath}")
        
        return True
