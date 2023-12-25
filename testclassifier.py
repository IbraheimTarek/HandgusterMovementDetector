import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage import exposure
from skimage.feature import hog
from skimage.filters import gaussian
# Function to compute HOG features
def compute_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred_image = gaussian(gray, sigma=0.01)

    # Apply histogram equalization for contrast enhancement
    #enhanced_image = exposure.equalize_hist(blurred_image)
    #cv2.imshow('Original Frame gray', blurred_image)
    # Compute HOG features
    hog_features, hog_image = hog(blurred_image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

    return hog_features,hog_image


def compute_hog(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor()

    # Compute HOG features
    features = hog.compute(gray)

    return features

def predict_class_hog(hog_features, classifier):
    # Ensure hog_features is a 2D array
    hog_features = hog_features.reshape(1, -1)

    # Predict the class
    predicted_class = classifier.predict(hog_features)
    return predicted_class[0]

# Load the trained HOG classifier
svm_classifier = joblib.load('trained_classifier_V9.pkl')

# Open an image or a video file for testing
# For testing with an image:
# img_path = 'path/to/your/test/image.jpg'
# frame = cv2.imread(img_path)

# For testing with a video file:
# cap = cv2.VideoCapture('path/to/your/test/video.mp4')

# For real-time testing using a camera:
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a different index

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame")
        break

    # Resize the frame to match the classifier's input size
    resized_frame = cv2.resize(frame, (320, 180))

    # Compute HOG features for the resized frame
    hog_features, hog_image = compute_hog_features(resized_frame)

    # Predict the class using HOG features
    orientation_prediction = predict_class_hog(hog_features,svm_classifier)

    # Display the predicted class on the frame
    cv2.putText(resized_frame, f'HOG Class: {orientation_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the HOG image
    cv2.imshow('HOG Features', hog_image)

    # Display the resized frame
    cv2.imshow('Resized Frame', resized_frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
