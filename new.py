import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
import joblib
import time

# Function to compute HOG features
def compute_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

# Load the trained SVM classifier
svm_classifier = joblib.load('trained_classifier_32bit.pkl')

# Create a background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera. Change it if you have multiple cameras.

# Set the desired width and height for the resized frame
desired_width = 32
desired_height = 32

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Resize the frame to a smaller size
    frame = cv2.resize(frame, (desired_width, desired_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)

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

    # Display the original and segmented frames along with the predicted orientation
    cv2.putText(segmented_hand, f'Orientation: {int(orientation_prediction[0])}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Segmented Hand', segmented_hand)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
