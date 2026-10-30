import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
import sys


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "models/hand_landmarker.task"
CSV_FILE = "dataset/isl_landmarks.csv"

# Full ISL alphabet
ALL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Number of samples for each sign
SAMPLES_PER_LABEL = 500


# --------------------------------------------------
# ISL SIGN DESCRIPTIONS
# --------------------------------------------------

SIGN_DESCRIPTIONS = {
    "A": "Closed fist, thumb on the side",
    "B": "Open palm, fingers together, thumb tucked",
    "C": "Curved hand forming C shape",
    "D": "Index finger up, others touch thumb",
    "E": "Fingers curled down, thumb tucked",
    "F": "Index-thumb circle, other fingers spread",
    "G": "Index finger pointing sideways, thumb out",
    "H": "Index and middle fingers pointing sideways",
    "I": "Pinky finger raised, fist closed",
    "J": "Pinky up then trace J curve down",
    "K": "Index and middle up, thumb between them",
    "L": "L shape with index finger and thumb",
    "M": "Three fingers over thumb, fist down",
    "N": "Two fingers over thumb, fist down",
    "O": "All fingers touch thumb forming O",
    "P": "Like K but pointing down",
    "Q": "Like G but pointing down",
    "R": "Index and middle crossed",
    "S": "Closed fist, thumb over fingers",
    "T": "Thumb between index and middle finger",
    "U": "Index and middle together pointing up",
    "V": "Index and middle spread (peace sign)",
    "W": "Index, middle, ring spread pointing up",
    "X": "Index finger hooked/bent",
    "Y": "Thumb and pinky extended (hang loose)",
    "Z": "Index finger traces Z in air",
}


# --------------------------------------------------
# PARSE COMMAND LINE ARGUMENTS
# --------------------------------------------------

def parse_args():
    """Parse optional command line arguments."""
    labels_to_collect = ALL_LABELS[:]
    samples = SAMPLES_PER_LABEL

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--letters" and i + 1 < len(sys.argv):
            letters = sys.argv[i + 1].upper()
            labels_to_collect = [ch for ch in letters if ch in ALL_LABELS]
            i += 2

        elif arg == "--samples" and i + 1 < len(sys.argv):
            samples = int(sys.argv[i + 1])
            i += 2

        elif arg == "--resume":
            # Automatically skip letters that already have enough samples
            labels_to_collect = get_remaining_labels(samples)
            i += 1

        elif arg == "--help":
            print()
            print("ISL Data Collection Tool")
            print("========================")
            print()
            print("Usage:")
            print("  python src/collect_data.py")
            print("  python src/collect_data.py --letters FGHIJ")
            print("  python src/collect_data.py --samples 300")
            print("  python src/collect_data.py --resume")
            print()
            print("Options:")
            print("  --letters XYZ   Collect only specified letters")
            print("  --samples N     Number of samples per letter (default: 500)")
            print("  --resume        Skip letters that already have enough samples")
            print("  --help          Show this help message")
            print()
            sys.exit(0)

        else:
            i += 1

    return labels_to_collect, samples


def get_remaining_labels(target_count):
    """Check CSV and return labels that need more samples."""
    remaining = []

    if not os.path.exists(CSV_FILE):
        return ALL_LABELS[:]

    # Count existing samples
    counts = {}
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        header = next(reader, None)

        if header is None:
            return ALL_LABELS[:]

        for row in reader:
            if len(row) >= 64:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1

    for label in ALL_LABELS:
        current = counts.get(label, 0)

        if current < target_count:
            remaining.append(label)

    return remaining


# --------------------------------------------------
# CREATE DATASET FOLDER
# --------------------------------------------------

os.makedirs("dataset", exist_ok=True)


# --------------------------------------------------
# CREATE CSV FILE IF NEEDED
# --------------------------------------------------

def ensure_csv_exists():
    """Create CSV file with header if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        header = []

        # 21 landmarks x 3 coordinates = 63 features
        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")
            header.append(f"z{i}")

        # Add the class label
        header.append("label")

        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)


# --------------------------------------------------
# COUNT EXISTING SAMPLES
# --------------------------------------------------

def count_existing_samples():
    """Count how many samples already exist per label."""
    counts = {}

    if not os.path.exists(CSV_FILE):
        return counts

    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) >= 64:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1

    return counts


# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# NORMALIZE LANDMARKS
# --------------------------------------------------

def normalize_landmarks(hand_landmarks):

    landmarks = []

    # Get wrist coordinates
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    # Make wrist the reference point
    for landmark in hand_landmarks:

        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z

        landmarks.extend([x, y, z])

    # Convert to NumPy array
    landmarks = np.array(landmarks, dtype=np.float32)

    # Scale the values
    max_value = np.max(np.abs(landmarks))

    if max_value > 0:
        landmarks = landmarks / max_value

    return landmarks.tolist()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    labels_to_collect, samples_per_label = parse_args()

    if not labels_to_collect:
        print()
        print("All letters already have enough samples!")
        print("Use --help for options.")
        return

    ensure_csv_exists()
    existing_counts = count_existing_samples()

    # Start camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    print()
    print("=" * 50)
    print("   ISL DATA COLLECTION SYSTEM")
    print("   Full Alphabet (A-Z)")
    print("=" * 50)
    print()

    print(f"Letters to collect: {', '.join(labels_to_collect)}")
    print(f"Samples per letter: {samples_per_label}")
    print()

    # Show existing data
    if existing_counts:
        print("Existing samples:")
        for label in ALL_LABELS:
            count = existing_counts.get(label, 0)
            if count > 0:
                print(f"  {label}: {count}")
        print()

    print("Controls:")
    print("  SPACE  = Start collecting current letter")
    print("  N      = Skip to next letter")
    print("  Q      = Quit and save")
    print()

    start_time = time.monotonic()
    total_collected = 0

    # --------------------------------------------------
    # COLLECT EACH SIGN
    # --------------------------------------------------

    for idx, label in enumerate(labels_to_collect):

        existing = existing_counts.get(label, 0)
        needed = max(0, samples_per_label - existing)

        if needed == 0:
            print(f"Skipping {label}: already has {existing} samples")
            continue

        description = SIGN_DESCRIPTIONS.get(label, "")

        print()
        print("-" * 50)
        print(f"  Letter {idx + 1}/{len(labels_to_collect)}: {label}")
        print(f"  Sign: {description}")
        print(f"  Need: {needed} samples (have {existing})")
        print("-" * 50)
        print("Show the sign and press SPACE when ready.")

        ready = False

        while not ready:
            success, frame = camera.read()
            if not success:
                print("Could not access camera.")
                break

            frame = cv2.flip(frame, 1)

            # Display instructions
            cv2.putText(
                frame,
                f"Prepare Sign: {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 3
            )

            cv2.putText(
                frame,
                f"({description})",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2
            )

            cv2.putText(
                frame,
                f"Letter {idx + 1}/{len(labels_to_collect)}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            cv2.putText(
                frame,
                "SPACE=Start  N=Skip  Q=Quit",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2
            )

            cv2.imshow("ISL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                ready = True

            elif key == ord("n"):
                print(f"  Skipped {label}")
                break

            elif key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                detector.close()
                print()
                print(f"Quit. Total new samples collected: {total_collected}")
                return

        if not ready:
            continue

        # --------------------------------------------------
        # COLLECT SAMPLES
        # --------------------------------------------------

        samples_collected = 0

        while samples_collected < needed:

            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            # Check whether a hand was detected
            if result.hand_landmarks:

                hand_landmarks = result.hand_landmarks[0]

                # Normalize landmarks
                features = normalize_landmarks(hand_landmarks)

                # Save features + label
                with open(CSV_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(features + [label])

                samples_collected += 1
                total_collected += 1

                # Draw landmarks
                height, width, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Display progress
                progress = samples_collected / needed
                bar_width = 300
                bar_x = 30
                bar_y = 140

                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + 25),
                    (50, 50, 50), -1
                )

                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + int(bar_width * progress), bar_y + 25),
                    (0, 255, 0), -1
                )

                cv2.putText(
                    frame,
                    f"Sign: {label}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3
                )

                cv2.putText(
                    frame,
                    f"Samples: {samples_collected}/{needed}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Total: {total_collected}",
                    (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2
                )

            else:
                cv2.putText(
                    frame,
                    "Hand not detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Sign: {label} - {description}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2
                )

            cv2.imshow("ISL Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                detector.close()
                print()
                print(f"Quit. Total new samples: {total_collected}")
                return

        print(
            f"  Completed {label}: "
            f"{samples_collected} new samples"
        )

    # --------------------------------------------------
    # CLEAN UP
    # --------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()
    detector.close()

    print()
    print("=" * 50)
    print("Data collection completed!")
    print(f"Total new samples collected: {total_collected}")
    print(f"Dataset saved to: {CSV_FILE}")
    print("=" * 50)

    # Show final summary
    final_counts = count_existing_samples()
    print()
    print("Final dataset summary:")
    for label in ALL_LABELS:
        count = final_counts.get(label, 0)
        status = "OK" if count >= samples_per_label else "NEED MORE"
        print(f"  {label}: {count:>5} samples  [{status}]")


if __name__ == "__main__":
    main()