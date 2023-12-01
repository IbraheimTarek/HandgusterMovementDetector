import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
import joblib

import time

def compute_hog_features(image):
    # Convert the image to grayscale if it has more than 2 dimensions
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply HOG algorithm
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    
    return features

# Load the trained SVM classifier
svm_classifier = joblib.load('trained_classifier_32bit_2.pkl')

# Create a background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Lucas-Kanade parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera. Change it if you have multiple cameras.

# Set the desired width and height for the resized frame
desired_width = 32
desired_height = 32

# Initialize variables for optical flow
old_gray = None
p0 = None
no_movement_threshold = 0.2  # Adjust this threshold as needed

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Resize the frame to a smaller size
    frame = cv2.resize(frame, (desired_width, desired_height))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame_gray)

    # Remove noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Bitwise AND the original frame with the foreground mask to get the segmented hand
    segmented_hand = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Compute HOG features for the segmented hand
    hog_features = compute_hog_features(segmented_hand)

    # Reshape the features to match the expected input shape for the SVM classifier
    hog_features = hog_features.reshape(1, -1)

    # Predict hand orientation using the trained SVM classifier
    orientation_prediction = svm_classifier.predict(hog_features)

    # Apply Optical Flow
    if old_gray is not None and p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Calculate the average motion vector
        if len(good_new) > 0:
            average_motion_vector = np.mean(good_new - good_old, axis=0)

            # Determine movement direction based on the components of the motion vector
            if np.abs(average_motion_vector[0]) > np.abs(average_motion_vector[1]):
                if average_motion_vector[0] > no_movement_threshold:
                    print("Movement Direction: Right")
                elif average_motion_vector[0] < -no_movement_threshold:
                    print("Movement Direction: Left")
                else:
                    print("No Operation (Not moving horizontally)")
            else:
                if average_motion_vector[1] > no_movement_threshold:
                    print("Movement Direction: Down")
                elif average_motion_vector[1] < -no_movement_threshold:
                    print("Movement Direction: Up")
                else:
                    print("No Operation (Not moving vertically)")

        # Update p0 for the next frame
        p0 = good_new.reshape(-1, 1, 2)

    # Initialize p0 for the first frame
    if p0 is None:
        mask = np.zeros_like(frame_gray)
        mask[fg_mask > 0] = 255
        corners = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=10)
        p0 = corners.reshape(-1, 1, 2)

    # Show the segmented hand with optical flow
    cv2.imshow('Segmented Hand', segmented_hand)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update old_gray and p0 for the next iteration
    old_gray = frame_gray.copy()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
