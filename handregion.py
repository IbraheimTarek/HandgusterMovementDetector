import cv2
import numpy as np

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the frame
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or other criteria
    min_area_threshold = 100  # Adjust this value based on your requirements
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]

    # Create a binary mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise AND operation to get the hand region
    hand_region = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original and cropped frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Hand Region', hand_region)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
