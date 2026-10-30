import cv2

# Open the default camera
camera = cv2.VideoCapture(0)

while True:
    # Read one frame from the camera
    success, frame = camera.read()

    if not success:
        print("Could not access the camera.")
        break

    # Display the camera frame
    cv2.imshow("Camera Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()