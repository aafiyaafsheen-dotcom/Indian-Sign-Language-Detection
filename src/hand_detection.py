import cv2
import mediapipe as mp
import time

# Location of the MediaPipe hand model
MODEL_PATH = "models/hand_landmarker.task"

# MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure the hand detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create the hand detector
detector = HandLandmarker.create_from_options(options)

# Open the webcam
camera = cv2.VideoCapture(0)

# Used to create timestamps for MediaPipe
start_time = time.monotonic()

while True:

    # Read a frame
    success, frame = camera.read()

    if not success:
        print("Could not access the camera.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Create timestamp
    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    # Detect hand
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # Check if a hand was detected
    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            height, width, _ = frame.shape

            # Store landmark coordinates
            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

            # Draw the 21 landmarks
            for x, y in points:

                cv2.circle(
                    frame,
                    (x, y),
                    6,
                    (0, 0, 255),
                    -1
                )

            # Display number of landmarks
            cv2.putText(
                frame,
                "Hand Detected - 21 Landmarks",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Display the result
    cv2.imshow("Indian Sign Language - Hand Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
camera.release()
cv2.destroyAllWindows()
detector.close()