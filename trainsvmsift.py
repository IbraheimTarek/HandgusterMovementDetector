import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Function to compute SIFT features
def compute_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # Ensure that descriptors have the same length (zero-padding if needed)
    if descriptors is not None:
        descriptors = descriptors.flatten()
        # Set the maximum expected length (adjust as needed)
        max_length = 128 * 128  # Adjust this value based on your requirements
        descriptors = np.pad(descriptors, (0, max_length - len(descriptors)), 'constant')
    return descriptors

def load_images_from_folder(root_folder):
    images = []
    labels = []
    label_mapping = {}  # Map subfolder names to unique numeric labels

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    if subfolder_name not in label_mapping:
                        label_mapping[subfolder_name] = len(label_mapping)
                    label = label_mapping[subfolder_name]

                    for filename in os.listdir(subfolder_path):
                        img_path = os.path.join(subfolder_path, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            images.append(img)
                            labels.append(label)

    return images, labels

# Specify the root path to the dataset
root_dataset_path = 'traindataset'

# Load images and labels from the dataset
all_images, all_labels = load_images_from_folder(root_dataset_path)

# Combine positive and negative data
images = np.array(all_images)
labels = np.array(all_labels)

# Extract SIFT features for each image
sift_features = [compute_sift_features(img) for img in images]

# Convert SIFT features to numpy array
sift_features_np = np.array(sift_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sift_features_np, labels, test_size=0.2, random_state=42)

# Training the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Save the trained classifier
joblib.dump(svm_classifier, 'trained_classifier_sift_only.pkl')

# Evaluate the classifier on the test set
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
