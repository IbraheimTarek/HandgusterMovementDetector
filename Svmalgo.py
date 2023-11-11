import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Ensure the image has 8-bit depth
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate HOG features
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

# Function to train an SVM classifier
def train_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

# Function to predict hand direction using the trained SVM
def predict_hand_direction(svm_model, image):
    features = extract_hog_features(image)
    prediction = svm_model.predict([features])
    return prediction[0]

# Generate a synthetic dataset (replace this with your actual dataset)
# In a real scenario, you would load your images and corresponding labels.
# For simplicity, let's assume two classes: 0 for "Left" and 1 for "Right."
np.random.seed(42)
num_samples = 200
X_left = np.random.rand(num_samples, 64, 64, 3) * 255  # Replace with actual left-hand images
X_right = np.random.rand(num_samples, 64, 64, 3) * 255  # Replace with actual right-hand images
y_left = np.zeros(num_samples)
y_right = np.ones(num_samples)

# Combine left and right samples
X = np.vstack((X_left, X_right))
y = np.hstack((y_left, y_right))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract HOG features for each image
X_train_features = np.array([extract_hog_features(img) for img in X_train])
X_test_features = np.array([extract_hog_features(img) for img in X_test])

# Train SVM classifier
svm_classifier = train_svm(X_train_features, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_features)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Now, you can use the trained SVM to predict hand direction in real-time (replace with actual video capture)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the training data size
    frame = cv2.resize(frame, (64, 64))

    # Predict hand direction
    direction = predict_hand_direction(svm_classifier, frame)

    # Print the predicted direction
    print(f"Direction: {'Left' if direction == 0 else 'Right'}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
