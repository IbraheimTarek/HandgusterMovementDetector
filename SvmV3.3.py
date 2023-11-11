import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

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
X_up = np.random.rand(num_samples, 64, 64, 3) * 255 
X_down = np.random.rand(num_samples, 64, 64, 3) * 255  

y_left = np.zeros(num_samples)
y_right = np.ones(num_samples)
y_up = np.full(num_samples, 2)
y_down = np.full(num_samples, 3)

X = np.vstack((X_left, X_right, X_up, X_down))
y = np.hstack((y_left, y_right, y_up, y_down))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_features = np.array([extract_hog_features(img) for img in X_train])
X_test_features = np.array([extract_hog_features(img) for img in X_test])
svm_classifier = train_svm(X_train_features, y_train)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (64, 64))

    direction = predict_hand_direction(svm_classifier, frame)

    if direction == 0:
        print("Direction: Left")
    elif direction == 1:
        print("Direction: Right")
    elif direction == 2:
        print("Direction: Up")
    elif direction == 3:
        print("Direction: Down")

    cv2.imshow("Hand Movement Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
