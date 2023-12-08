import cv2
import numpy as np

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise and perform morphological operations
    fg_mask = cv2.medianBlur(fg_mask, 5)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the bounding box of the largest contour (assuming it's the hand)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Expand the bounding box to take a larger part of the image
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1], w + 2 * padding)
        h = min(frame.shape[0], h + 2 * padding)

        # Create a black canvas of the same size as the original frame
        canvas = np.zeros_like(frame)

        # Crop the hand region and paste it onto the black canvas
        hand_region = frame[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = hand_region

        # Display the original and cropped frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Hand Region', canvas)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
