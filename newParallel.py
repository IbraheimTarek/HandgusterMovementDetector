import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
import joblib
import time
import pyautogui
import concurrent.futures

def compute_hog_features(image):
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def process_frame(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fg_mask = background_subtractor.apply(frame_gray)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    segmented_hand = cv2.bitwise_and(frame, frame, mask=fg_mask)

    hog_features = compute_hog_features(frame)
    hog_features = hog_features.reshape(1, -1)

    orientation_prediction = svm_classifier.predict(hog_features)

    return frame_gray, fg_mask, segmented_hand, orientation_prediction

def output_direction(good_new, good_old,orientation_prediction):
    if len(good_new) > 0:
        average_motion_vector = np.mean(good_new - good_old, axis=0)

        if np.abs(average_motion_vector[0]) > np.abs(average_motion_vector[1]):
            if average_motion_vector[0] > no_movement_threshold and orientation_prediction:
                print("Movement Direction: Right")
                pyautogui.press('right')
            elif average_motion_vector[0] < -no_movement_threshold and orientation_prediction:
                print("Movement Direction: Left")
                pyautogui.press('left')
            else:
                print("No Operation (Not moving horizontally)")
        else:
            if average_motion_vector[1] > no_movement_threshold and orientation_prediction:
                print("Movement Direction: Down")
                pyautogui.press('down')
            elif average_motion_vector[1] < -no_movement_threshold and orientation_prediction:
                print("Movement Direction: Up")
                pyautogui.press('up')
            else:
                print("No Operation (Not moving vertically)")

# Load the trained SVM classifier
svm_classifier = joblib.load('trained_classifier_64bit.pkl')

# Create a background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Set the desired width and height for the resized frame
desired_width = 64
desired_height = 64

# Initialize variables for optical flow
old_gray = None
p0 = None
no_movement_threshold = 0.2  # Adjust this threshold as needed

# Lucas-Kanade parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

with concurrent.futures.ThreadPoolExecutor() as executor:
    while True:
        start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        frame = cv2.resize(frame, (desired_width, desired_height))

        future = executor.submit(process_frame, frame)

        # Retrieve the results of the processing
        frame_gray, fg_mask, segmented_hand, orientation_prediction = future.result()

        # Apply Optical Flow
        if old_gray is not None and p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Parallelize the computation of average motion vector and decision-making
            if(orientation_prediction):
                executor.submit(output_direction, good_new, good_old,orientation_prediction)

            p0 = good_new.reshape(-1, 1, 2)

        if p0 is None:
            mask = np.zeros_like(frame_gray)
            mask[fg_mask > 0] = 255
            corners = cv2.goodFeaturesToTrack(frame_gray, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=10)
            p0 = corners.reshape(-1, 1, 2)

        cv2.imshow('Segmented Hand', segmented_hand)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        old_gray = frame_gray.copy()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()