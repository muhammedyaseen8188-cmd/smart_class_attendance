# train_model.py
"""
Train the face recognition model using deep embeddings.
Uses the face_recognition library (dlib CNN backend, 99.38% accuracy).

Run after augmenting: python train_model.py
"""
import face_recognition
import os
import pickle
import numpy as np
from pathlib import Path


def train_from_directory(faces_dir="known_faces_augmented",
                          output_path="face_encodings.pkl"):
    """
    Load all images from faces_dir, compute 128-dim face embeddings,
    and save them as a dictionary mapping person_name → [list of encodings].
    
    This replaces the LBPH .yml model file entirely.
    One encoding is a 128-dimensional float vector — mathematically
    unique to a face, like a fingerprint. Recognition works by finding
    the stored encoding closest to the live encoding.
    """
    faces_path = Path(faces_dir)
    
    if not faces_path.exists():
        print(f"ERROR: Directory '{faces_dir}' not found.")
        print("Run python augment_faces.py first.")
        return False

    known_encodings = {}   # person_name → list of 128-dim vectors
    failed_images = []

    print(f"Loading faces from '{faces_dir}'...")
    
    for person_dir in sorted(faces_path.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        encodings_for_person = []

        image_files = [f for f in person_dir.iterdir()
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        print(f"\n  Processing: {person_name} ({len(image_files)} images)")

        for img_file in image_files:
            # face_recognition uses RGB (not BGR like OpenCV)
            image = face_recognition.load_image_file(str(img_file))

            # Detect face locations in the image
            # model="cnn" is more accurate but slower; "hog" is faster
            # Use "hog" for training (speed), "cnn" for live recognition if GPU available
            face_locations = face_recognition.face_locations(image, model="hog")

            if len(face_locations) == 0:
                failed_images.append(str(img_file))
                continue

            if len(face_locations) > 1:
                # Take the largest face if multiple detected
                face_locations = [max(face_locations,
                                      key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))]

            # Compute the 128-dimensional encoding for this face
            # num_jitters=1 applies slight distortions during encoding for robustness
            encodings = face_recognition.face_encodings(
                image, face_locations, num_jitters=1
            )

            if encodings:
                encodings_for_person.append(encodings[0])

        if encodings_for_person:
            known_encodings[person_name] = encodings_for_person
            print(f"    ✓ {len(encodings_for_person)} valid encodings stored")
        else:
            print(f"    ✗ NO valid encodings found for {person_name}!")

    if not known_encodings:
        print("\nERROR: No encodings generated. Check your images.")
        return False

    # Save the encodings dictionary
    with open(output_path, "wb") as f:
        pickle.dump(known_encodings, f)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"  People enrolled: {len(known_encodings)}")
    print(f"  Total encodings: {sum(len(v) for v in known_encodings.values())}")
    print(f"  Failed images:   {len(failed_images)}")
    print(f"  Model saved to:  {output_path}")
    print(f"{'='*50}")

    if failed_images:
        print(f"\nImages with no face detected:")
        for f in failed_images[:10]:
            print(f"  - {f}")

    return True


if __name__ == "__main__":
    train_from_directory()            