import cv2
import numpy as np

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Set the desired frame width and height
frame_width = 320
frame_height = 180

# Set the frame width and height for the camera
cap.set(3, frame_width)
cap.set(4, frame_height)

# Initialize the previous frame
ret, gray_previous = cap.read()
gray_previous = cv2.cvtColor(gray_previous, cv2.COLOR_BGR2GRAY)

# Parameters for optical flow calculation
param = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray_previous, gray, None, **param)

    # Calculate magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold for motion detection
    threshold = 2.0

    # Create a binary mask based on the magnitude threshold
    mask = np.zeros_like(gray)
    mask[magnitude > threshold] = 255

    # Apply the mask to the original frame
    moving_objects = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Moving Objects', moving_objects)

    # Update the previous frame
    gray_previous = gray

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
