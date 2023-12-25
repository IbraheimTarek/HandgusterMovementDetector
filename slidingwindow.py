import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage import exposure

def compute_hog_features(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(_, in_range=(0, 10))
    return features, hog_image_rescaled

def predict_class_hog(hog_features, classifier):
    # Ensure hog_features is a 1D array
    hog_features = hog_features.flatten()

    # Predict the class
    predicted_class = classifier.predict(hog_features.reshape(1, -1))
    return predicted_class[0]

svm_classifier = joblib.load('trained_classifier_V3.pkl')

cap = cv2.VideoCapture(0)

# Set the sliding window size and step
window_size = (128, 128)
step_size = 20

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 180))
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the size of the frame
    height, width = gray_frame.shape

    # Create an array of shape (height, width) with a step of step_size
    y_range = np.arange(0, height - window_size[1], step_size)
    x_range = np.arange(0, width - window_size[0], step_size)

    # Create a 2D grid of indices
    y_indices, x_indices = np.meshgrid(y_range, x_range)

    # Iterate over the grid and extract ROIs using slicing
    for y, x in zip(y_indices.flatten(), x_indices.flatten()):
        roi = gray_frame[y:y + window_size[1], x:x + window_size[0]]

        # Compute HOG features for the ROI
        hog_features, _ = compute_hog_features(roi)

        # Predict the class using HOG features
        orientation_prediction = predict_class_hog(hog_features, svm_classifier)

        # Draw a rectangle around the window only if the hand is detected
        if orientation_prediction == 1:  # Assuming class 1 corresponds to the hand
            cv2.rectangle(frame, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

    cv2.imshow('Hand Detection with Sliding Window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
