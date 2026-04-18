# augment_faces.py
"""
Data augmentation utility — expands 3 training images per student to 30+.
Run this ONCE before training your model:
    python augment_faces.py
"""
import cv2
import os
import numpy as np
import random
from pathlib import Path


def augment_face_image(image):
    """
    Generate multiple augmented variants of a single face image.
    Each variant simulates a real-world condition: different lighting,
    slight angle, brightness, blur, etc.
    
    Returns a list of augmented images.
    """
    augmented = []
    h, w = image.shape[:2]

    # 1. Original (always include)
    augmented.append(image.copy())

    # 2. Horizontal flip (simulates looking slightly left vs right)
    augmented.append(cv2.flip(image, 1))

    # 3. Brightness variations (simulates different room lighting)
    for gamma in [0.6, 0.8, 1.3, 1.6]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        augmented.append(cv2.LUT(image, table))

    # 4. Rotation (slight head tilt ±5° and ±10°)
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    # 5. Gaussian noise (simulates low-light grain)
    for _ in range(2):
        noise = np.random.normal(0, 12, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)

    # 6. Slight blur (simulates camera out of focus)
    augmented.append(cv2.GaussianBlur(image, (5, 5), 0))
    augmented.append(cv2.GaussianBlur(image, (3, 3), 0))

    # 7. Contrast adjustment
    for alpha in [0.75, 1.25]:
        adjusted = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        augmented.append(adjusted)

    # 8. Slight crop and resize (simulates face at different distances)
    for crop_pct in [0.85, 0.90]:
        margin_x = int(w * (1 - crop_pct) / 2)
        margin_y = int(h * (1 - crop_pct) / 2)
        cropped = image[margin_y:h-margin_y, margin_x:w-margin_x]
        resized = cv2.resize(cropped, (w, h))
        augmented.append(resized)

    return augmented


def augment_known_faces_directory(known_faces_dir="known_faces",
                                   augmented_dir="known_faces_augmented"):
    """
    Read every image from known_faces/, generate augmented variants,
    and save them to known_faces_augmented/.
    
    The augmented directory is what the model trains on.
    The original known_faces/ directory remains untouched.
    """
    known_path = Path(known_faces_dir)
    aug_path = Path(augmented_dir)

    if not known_path.exists():
        print(f"ERROR: '{known_faces_dir}' directory not found.")
        return

    total_original = 0
    total_augmented = 0

    for person_dir in known_path.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        aug_person_dir = aug_path / person_name
        aug_person_dir.mkdir(parents=True, exist_ok=True)

        images_for_person = 0

        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  WARNING: Could not load {img_file}")
                continue

            total_original += 1
            images_for_person += 1

            # Generate augmented variants
            variants = augment_face_image(image)

            # Save each variant
            stem = img_file.stem
            for idx, variant in enumerate(variants):
                save_path = aug_person_dir / f"{stem}_aug{idx:02d}.jpg"
                cv2.imwrite(str(save_path), variant)
                total_augmented += 1

        print(f"  {person_name}: {images_for_person} originals → {images_for_person * 18} augmented")

    print(f"\nDone! {total_original} original images → {total_augmented} augmented images")
    print(f"Augmented data saved to: {aug_path.resolve()}")


if __name__ == "__main__":
    augment_known_faces_directory()