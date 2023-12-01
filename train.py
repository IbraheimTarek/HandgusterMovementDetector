import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import joblib
import os

# Function to compute HOG features
def compute_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def load_images_from_folder(folder):
    images = []
    image_paths = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            image_paths.append(img_path)
    return images, image_paths

# Specify the path to the "train" folder
train_folder = 'train'

all_images, all_image_paths = load_images_from_folder(train_folder)

# Positive images are assumed to contain hands, and negative images are assumed to be non-hand objects or backgrounds.
positive_images = [img for img, path in zip(all_images, all_image_paths) if 'positive' in os.path.basename(path)]
negative_images = [img for img, path in zip(all_images, all_image_paths) if 'negative' in os.path.basename(path)]

# Check if there are images in both classes
if not positive_images or not negative_images:
    print("Error: Insufficient data. Ensure there are images in both positive and negative classes.")
    exit()

positive_labels = np.ones(len(positive_images))
negative_labels = np.zeros(len(negative_images))

# Combine positive and negative data
images = positive_images + negative_images
labels = np.concatenate([positive_labels, negative_labels])

# Extract HOG features for each image
hog_features = [compute_hog_features(img) for img in images]
if len(images) == 0 or len(labels) == 0:
    print("Error: Insufficient data. Ensure there are images in both positive and negative classes.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, train_image_paths, test_image_paths = train_test_split(hog_features, labels, all_image_paths, test_size=0.2, random_state=42)

if len(X_train) == 0 or len(y_train) == 0:
    print("Error: Insufficient data. Ensure there are images in both positive and negative classes.")
    exit()

# Training --> the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Save the trained classifier and change the name upon using a different image size other than in the train folder
joblib.dump(svm_classifier, 'trained_classifier_64bit.pkl')

# Evaluate the classifier on the test set
y_pred = svm_classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Print image names in the train set
print("Images in Train Set:")
for img_path in train_image_paths:
    print(os.path.basename(img_path))

# Print image names in the test set
print("\nImages in Test Set:")
for img_path in test_image_paths:
    print(os.path.basename(img_path))
