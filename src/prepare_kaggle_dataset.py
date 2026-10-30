"""
Prepare ISL Dataset from Image Folders

This script extracts MediaPipe hand landmarks from ISL alphabet
image datasets (e.g., from Kaggle) and saves them as a CSV file
in the same format used by the project.

Expected input structure:
    dataset/raw/A/*.jpg (or .png)
    dataset/raw/B/*.jpg
    ...
    dataset/raw/Z/*.jpg

Output:
    dataset/isl_landmarks.csv

Usage:
    python src/prepare_kaggle_dataset.py
    python src/prepare_kaggle_dataset.py --input dataset/raw
    python src/prepare_kaggle_dataset.py --max-per-class 500
"""

import cv2
import mediapipe as mp
import csv
import os
import sys
import numpy as np
from pathlib import Path


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "models/hand_landmarker.task"
DEFAULT_INPUT = "dataset/raw"
CSV_FILE = "dataset/isl_landmarks.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# PARSE ARGUMENTS
# --------------------------------------------------

def parse_args():

    input_dir = DEFAULT_INPUT
    max_per_class = None
    output_file = CSV_FILE

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--input" and i + 1 < len(sys.argv):
            input_dir = sys.argv[i + 1]
            i += 2

        elif arg == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2

        elif arg == "--max-per-class" and i + 1 < len(sys.argv):
            max_per_class = int(sys.argv[i + 1])
            i += 2

        elif arg == "--help":
            print()
            print("ISL Dataset Preparation Tool")
            print("============================")
            print()
            print("Extracts MediaPipe hand landmarks from image folders.")
            print()
            print("Expected input structure:")
            print("  dataset/raw/A/*.jpg")
            print("  dataset/raw/B/*.jpg")
            print("  ...")
            print()
            print("Options:")
            print("  --input DIR          Input directory (default: dataset/raw)")
            print("  --output FILE        Output CSV file (default: dataset/isl_landmarks.csv)")
            print("  --max-per-class N    Max samples per class")
            print("  --help               Show this help message")
            print()
            sys.exit(0)

        else:
            i += 1

    return input_dir, output_file, max_per_class


# --------------------------------------------------
# NORMALIZE LANDMARKS
# --------------------------------------------------

def normalize_landmarks(hand_landmarks):
    """Normalize hand landmarks using wrist as reference."""

    landmarks = []

    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    for landmark in hand_landmarks:
        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z
        landmarks.extend([x, y, z])

    landmarks = np.array(landmarks, dtype=np.float32)

    max_value = np.max(np.abs(landmarks))

    if max_value > 0:
        landmarks = landmarks / max_value

    return landmarks.tolist()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    input_dir, output_file, max_per_class = parse_args()

    print()
    print("=" * 50)
    print("ISL DATASET PREPARATION")
    print("=" * 50)
    print()

    # Check input directory
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        print()
        print("Please download an ISL image dataset and extract it to:")
        print(f"  {input_dir}/A/")
        print(f"  {input_dir}/B/")
        print(f"  ...")
        print(f"  {input_dir}/Z/")
        print()
        print("Recommended datasets:")
        print("  - Kaggle: prathumarikeri/indian-sign-language-isl")
        print("  - Kaggle: aryantiwari123/isl-dataset")
        return

    # Find class directories
    class_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
        and len(d) == 1
        and d.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ])

    if not class_dirs:
        # Try numbered or differently named directories
        all_dirs = sorted(os.listdir(input_dir))
        print(f"Found directories: {all_dirs[:10]}...")
        print("ERROR: No single-letter class directories found (A, B, C, ...)")
        return

    print(f"Found {len(class_dirs)} class directories: {', '.join(class_dirs)}")
    print()

    # Setup MediaPipe (IMAGE mode for static images)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )

    detector = HandLandmarker.create_from_options(options)

    # Create CSV header
    header = []
    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")
    header.append("label")

    # Process images
    total_processed = 0
    total_detected = 0
    class_counts = {}

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for class_dir in class_dirs:
            label = class_dir.upper()
            class_path = os.path.join(input_dir, class_dir)

            # Get image files
            images = [
                f for f in os.listdir(class_path)
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS
            ]

            if max_per_class:
                images = images[:max_per_class]

            detected = 0
            processed = 0

            for img_file in images:
                img_path = os.path.join(class_path, img_file)

                try:
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_image
                    )

                    # Detect hand
                    result = detector.detect(mp_image)

                    processed += 1

                    if result.hand_landmarks:
                        hand_landmarks = result.hand_landmarks[0]
                        features = normalize_landmarks(hand_landmarks)
                        writer.writerow(features + [label])
                        detected += 1

                except Exception as e:
                    continue

            class_counts[label] = detected
            total_processed += processed
            total_detected += detected

            detection_rate = (detected / processed * 100) if processed > 0 else 0
            print(
                f"  {label}: {processed:>5} images processed, "
                f"{detected:>5} hands detected "
                f"({detection_rate:.1f}%)"
            )

    detector.close()

    # Summary
    print()
    print("=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)
    print()
    print(f"Total images processed: {total_processed}")
    print(f"Total landmarks extracted: {total_detected}")
    print(f"Detection rate: {total_detected/total_processed*100:.1f}%"
          if total_processed > 0 else "N/A")
    print(f"Output file: {output_file}")
    print()
    print("Samples per class:")

    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        print(f"  {label}: {count}")

    # Check balance
    counts = list(class_counts.values())
    if counts:
        min_count = min(counts)
        max_count = max(counts)

        if max_count > min_count * 1.5:
            print()
            print("WARNING: Significant class imbalance detected.")
            print(f"  Min: {min_count}, Max: {max_count}")
            print("  Consider balancing the dataset before training.")


if __name__ == "__main__":
    main()
