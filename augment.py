"""
augment.py  —  Dataset Augmentation for Vest Detection
=======================================================
Tailored for the vest-no-vest-1 dataset structure:

    vest-no-vest-1/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/

Classes:
    0 = no vest
    1 = vest

Run:
    python augment.py

Output:
    vest-no-vest-1-augmented/
    ├── train/
    │   ├── images/   ← original + augmented images
    │   └── labels/   ← matching labels
    └── valid/        ← copied as-is (never augment validation)

Requirements:
    pip install albumentations opencv-python tqdm
"""

import os
import cv2
import shutil
import random
import hashlib
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("[ERROR] albumentations not installed.")
    print("        Run: pip install albumentations")
    exit(1)


# =========================
# CONFIG
# =========================
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR        = os.path.join(BASE_DIR, "vest-no-vest-1")
OUTPUT_DIR         = os.path.join(BASE_DIR, "vest-no-vest-1-augmented")
AUGMENTS_PER_IMAGE = 4
SEED               = 42


# =========================
# AUGMENTATION PIPELINE
# =========================
transform = A.Compose([

    # Lighting variations (night shifts, indoor, shadows)
    A.RandomBrightnessContrast(
        brightness_limit=(-0.4, 0.3),
        contrast_limit=0.3,
        p=0.8
    ),
    A.RandomGamma(gamma_limit=(60, 140), p=0.5),

    # Shadow simulation
    A.RandomShadow(
        shadow_roi=(0, 0.3, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=3,
        shadow_dimension=5,
        p=0.4
    ),

    # Color shift (different vest colors, lighting tones)
    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=40,
        val_shift_limit=30,
        p=0.5
    ),

    # Blur and noise (camera quality, motion)
    A.OneOf([
        A.MotionBlur(blur_limit=7, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(var_limit=(10, 50), p=1.0),
    ], p=0.4),

    # Occlusion simulation (person behind wall/object)
    A.CoarseDropout(
        max_holes=4,
        max_height=80,
        max_width=80,
        min_holes=1,
        min_height=20,
        min_width=20,
        fill_value=0,
        p=0.35
    ),

    # Geometric (different camera angles)
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
    A.Perspective(scale=(0.03, 0.07), p=0.3),

    # Image quality (compressed/low-res feeds)
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.3
))


# =========================
# HELPERS
# =========================
def safe_filename(stem, suffix=""):
    """Shorten filename to avoid Windows 260-char path limit."""
    max_stem = 60
    if len(stem) > max_stem:
        h = hashlib.md5(stem.encode()).hexdigest()[:10]
        stem = stem[:max_stem] + "_" + h
    return stem + suffix


def read_yolo_labels(label_path):
    class_labels, bboxes = [], []
    if not os.path.exists(label_path):
        return class_labels, bboxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return class_labels, bboxes


def write_yolo_labels(label_path, class_labels, bboxes):
    with open(label_path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")


def copy_split(split_name, src_dir, dst_dir):
    src = os.path.join(src_dir, split_name)
    dst = os.path.join(dst_dir, split_name)
    if os.path.exists(src):
        shutil.copytree(src, dst)
        print(f"[INFO] Copied {split_name}/ unchanged → {dst}")
    else:
        print(f"[WARN] {split_name}/ not found, skipping.")


def get_image_files(images_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]


# =========================
# AUGMENT TRAIN SPLIT
# =========================
def augment_train(src_dir, dst_dir, augments_per_image):
    src_images = os.path.join(src_dir, "train", "images")
    src_labels = os.path.join(src_dir, "train", "labels")
    dst_images = os.path.join(dst_dir, "train", "images")
    dst_labels = os.path.join(dst_dir, "train", "labels")

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    image_files = get_image_files(src_images)
    if not image_files:
        print("[ERROR] No images found in train/images.")
        return 0, 0, 0

    print(f"\n[INFO] Found {len(image_files)} training images.")
    print(f"[INFO] Generating {augments_per_image} augmented copies each.")
    print(f"[INFO] Total output images: {len(image_files) * (augments_per_image + 1)}\n")

    original_count  = 0
    augmented_count = 0
    skipped_count   = 0

    for idx, img_file in enumerate(tqdm(image_files, desc="Augmenting")):
        img_path   = os.path.join(src_images, img_file)
        raw_stem   = os.path.splitext(img_file)[0]
        label_path = os.path.join(src_labels, raw_stem + ".txt")

        # Shorten long filenames
        safe_stem  = safe_filename(raw_stem, suffix=f"_{idx}")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Could not read {img_file}, skipping.")
            skipped_count += 1
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        class_labels, bboxes = read_yolo_labels(label_path)

        # Save original with safe name
        cv2.imwrite(os.path.join(dst_images, safe_stem + ".jpg"), image)
        write_yolo_labels(os.path.join(dst_labels, safe_stem + ".txt"), class_labels, bboxes)
        original_count += 1

        # Generate augmented copies
        for i in range(augments_per_image):
            try:
                augmented  = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                aug_image  = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                aug_name = f"{safe_stem}_aug{i}"
                cv2.imwrite(
                    os.path.join(dst_images, aug_name + ".jpg"),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                write_yolo_labels(
                    os.path.join(dst_labels, aug_name + ".txt"),
                    aug_labels, aug_bboxes
                )
                augmented_count += 1

            except Exception as e:
                print(f"\n[WARN] Failed for {img_file} aug{i}: {e}")

    return original_count, augmented_count, skipped_count


# =========================
# DATA.YAML
# =========================
def write_data_yaml(output_dir):
    yaml_content = f"""# Augmented dataset — auto-generated by augment.py
path: {output_dir}
train: train/images
val:   valid/images

nc: 2
names:
  - no vest
  - vest
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[INFO] data.yaml written → {yaml_path}")
    return yaml_path


# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 55)
    print("   Vest Detection — Dataset Augmentation")
    print("=" * 55)
    print(f"Source  : {DATASET_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Copies  : {AUGMENTS_PER_IMAGE} augmented versions per image")
    print("=" * 55)

    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset folder not found: {DATASET_DIR}")
        return

    # Clean old output
    if os.path.exists(OUTPUT_DIR):
        print(f"\n[WARN] Deleting old output folder...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Step 1 — Augment train
    print("\n[STEP 1] Augmenting training data...")
    original_count, augmented_count, skipped_count = augment_train(
        DATASET_DIR, OUTPUT_DIR, AUGMENTS_PER_IMAGE
    )

    # Step 2 — Copy valid unchanged
    print("\n[STEP 2] Copying validation data unchanged...")
    copy_split("valid", DATASET_DIR, OUTPUT_DIR)

    # Step 3 — Copy test if exists
    if os.path.exists(os.path.join(DATASET_DIR, "test")):
        print("\n[STEP 3] Copying test data unchanged...")
        copy_split("test", DATASET_DIR, OUTPUT_DIR)

    # Step 4 — Write data.yaml
    print("\n[STEP 4] Writing data.yaml...")
    yaml_path = write_data_yaml(OUTPUT_DIR)

    # Summary
    total = original_count + augmented_count
    print("\n" + "=" * 55)
    print("   AUGMENTATION COMPLETE")
    print("=" * 55)
    print(f"   Original images  : {original_count}")
    print(f"   Augmented images : {augmented_count}")
    print(f"   Skipped          : {skipped_count}")
    print(f"   Total train imgs : {total}")
    print(f"   Output folder    : {OUTPUT_DIR}")
    print(f"   data.yaml        : {yaml_path}")
    print("=" * 55)
    print("\n[NEXT STEP] Update train.py to use the new data.yaml")
    print(f"            data = r'{yaml_path}'")


if __name__ == "__main__":
    main()