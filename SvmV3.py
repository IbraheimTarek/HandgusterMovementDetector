import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

def extract_hand_landmarks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    landmarks = []
    if contours:
        contour = max(contours, key=cv2.contourArea)
        for point in contour:
            x, y = point[0]
            landmarks.append((x, y))

    return landmarks

def extract_hog_features(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def train_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

def predict_hand_direction(svm_model, image):
    features = extract_hog_features(image)
    prediction = svm_model.predict([features])
    return prediction[0]

np.random.seed(42)
num_samples = 200
X_left = np.random.rand(num_samples, 64, 64, 3) * 255
X_right = np.random.rand(num_samples, 64, 64, 3) * 255
y_left = np.zeros(num_samples)
y_right = np.ones(num_samples)
X = np.vstack((X_left, X_right))
y = np.hstack((y_left, y_right))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_features = np.array([extract_hog_features(img) for img in X_train])
X_test_features = np.array([extract_hog_features(img) for img in X_test])
svm_classifier = train_svm(X_train_features, y_train)

cap = cv2.VideoCapture(0)
prev_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (64, 64))
    landmarks = extract_hand_landmarks(frame)
    direction = predict_hand_direction(svm_classifier, frame)

    if prev_landmarks is not None and landmarks:
        # Ensure the number of landmarks is consistent
        min_len = min(len(prev_landmarks), len(landmarks))
        prev_landmarks = prev_landmarks[:min_len]
        landmarks = landmarks[:min_len]

        x_difference = landmarks[0][0] - prev_landmarks[0][0]
        y_difference = landmarks[0][1] - prev_landmarks[0][1]

        # Adjust thresholds dynamically based on the average hand size
        avg_hand_size = np.mean(np.sqrt(np.sum(np.square(np.array(landmarks) - np.array(prev_landmarks)), axis=1)))
        x_threshold = 0.5 * avg_hand_size
        y_threshold = 0.5 * avg_hand_size

        if abs(x_difference) > x_threshold:
            if x_difference > 0:
                print("Hand moving right")
            else:
                print("Hand moving left")
        elif abs(y_difference) > y_threshold:
            if y_difference > 0:
                print("Hand moving down")
            else:
                print("Hand moving up")

    # Update previous landmarks
    prev_landmarks = landmarks

    print(f"Direction: {'Left' if direction == 0 else 'Right'}")

    cv2.imshow("Hand Direction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
