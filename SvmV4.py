import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
# the best result till now
# Function to extract HOG features from an image
def extract_hog_features(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Convert the image to grayscale and apply histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Calculate HOG features
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

# Function to traaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaain an SVM classifier
#   :-:
def train_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

# Function to predict hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhand direction using the trained SVM
def predict_hand_direction(svm_model, hand_roi):
    features = extract_hog_features(hand_roi)
    prediction = svm_model.predict([features])
    return prediction[0]


# Function to dynamically adjust the region of interest (ROI) based on hand landmarks
def extract_hand_landmarks(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding to segment the hand from the background
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area and keep the largest one (assumed to be the hand)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.0005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        landmarks = [(center_x, center_y)]  # Center of the bounding box
        return landmarks, (x, y, w, h)
    return [], None



np.random.seed(42)
num_samples = 200
X_left = np.random.rand(num_samples, 64, 64, 3) * 255  # Replace with actual left-hand images
X_right = np.random.rand(num_samples, 64, 64, 3) * 255  # Replace with actual right-hand images
X_up = np.random.rand(num_samples, 64, 64, 3) * 255  # Replace with actual up-hand images
X_down = np.random.rand(num_samples, 64, 64, 3) * 255  # Replace with actual down-hand images

y_left = np.zeros(num_samples)
y_right = np.ones(num_samples)
y_up = np.full(num_samples, 2)
y_down = np.full(num_samples, 3)


X = np.vstack((X_left, X_right, X_up, X_down))
y = np.hstack((y_left, y_right, y_up, y_down))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Extract HOG features for each image
X_train_features = np.array([extract_hog_features(img) for img in X_train])
X_test_features = np.array([extract_hog_features(img) for img in X_test])
# Train SVM classifier
svm_classifier = train_svm(X_train_features, y_train)

cap = cv2.VideoCapture(0)
# Initialize variables for temporal smoothing
prev_directions = []


#Real Time looooooooopp
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the training data size
    frame = cv2.resize(frame, (64, 64))

    # Extract hand landmarks and dynamically adjust the region of interest (ROI)
    landmarks, roi = extract_hand_landmarks(frame)
    
    if roi is not None:
        x, y, w, h = roi
        hand_roi = frame[y:y+h, x:x+w]
        
        # Predict hand direction
        direction = predict_hand_direction(svm_classifier, hand_roi)

        # Append the current direction to the history
        prev_directions.append(direction)

        # Keep the history limited to a certain length
        history_length = 5
        if len(prev_directions) > history_length:
            prev_directions = prev_directions[-history_length:]

        # Calculate the most common direction in the recent history
        smoothed_direction = max(set(prev_directions), key=prev_directions.count)

        print(f"Direction: {'Left' if smoothed_direction == 0 else 'Right' if smoothed_direction == 1 else 'Up' if smoothed_direction == 2 else 'Down'}")
        #hand_roi_features = extract_hog_features(hand_roi)
        #print(len(hand_roi_features))

    cv2.imshow("Hand Direction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
