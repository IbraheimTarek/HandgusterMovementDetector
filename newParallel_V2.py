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
    
    _, thresholded = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_threshold = 100  # Adjust this value based on your requirements
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    fg_mask = np.zeros_like(frame_gray)
    cv2.drawContours(fg_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    hand_region = cv2.bitwise_and(frame, frame, mask=fg_mask)


    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = contours
    # Extract the bounding box of the largest contour (assuming it's the hand)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Expand the bounding box to take a larger part of the image
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1], w + 2 * padding)
        h = min(frame.shape[0], h + 2 * padding)

        # Create a black canvas of the same size as the original frame
        canvas = np.zeros_like(frame)

        # Crop the hand region and paste it onto the black canvas
        hand_region = frame[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = hand_region


    segmented_hand = cv2.bitwise_and(canvas, canvas, mask=fg_mask)

    hog_features = compute_hog_features(frame)
    hog_features = hog_features.reshape(1, -1)

    orientation_prediction = svm_classifier.predict(hog_features)
    #print(orientation_prediction)
    return frame_gray, fg_mask, segmented_hand, orientation_prediction

def calculate_hand_centroid(mask):
    
    # Find contours in the segmented hand region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the maximum area (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the hand contour
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Determine the hand movement direction based on centroid change
            if "prev_cx" in globals():
                dx = cx - prev_cx
                dy = cy - prev_cy

                if abs(dx) > abs(dy):
                    if dx > no_movement_threshold:
                        print("Movement Direction: Right")
                        pyautogui.press('right')
                    elif dx < -no_movement_threshold:
                        print("Movement Direction: Left")
                        pyautogui.press('left')
                else:
                    if dy > no_movement_threshold:
                        print("Movement Direction: Down")
                        pyautogui.press('down')
                    elif dy < -no_movement_threshold:
                        print("Movement Direction: Up")
                        pyautogui.press('up')

            # Update previous centroid
            globals()["prev_cx"], globals()["prev_cy"] = cx, cy

# Load the trained SVM classifier
svm_classifier = joblib.load('trained_knn_classifier_64bit_V2.pkl')

# Create a background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Set the desired width and height for the resized frame
desired_width = 64
desired_height = 64
no_movement_threshold = 0 
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

        # Calculate the centroid of the hand region
        if(orientation_prediction):
            center = executor.submit(calculate_hand_centroid, fg_mask) 

        cv2.imshow('Segmented Hand', fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
