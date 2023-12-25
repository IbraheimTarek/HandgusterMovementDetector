import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
import joblib
import os

# Function to compute LBP features
def compute_lbp_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred_image = gaussian(gray, sigma=0.01)

    # Compute LBP features
    lbp_radius = 3
    lbp_points = 8 * lbp_radius
    lbp = local_binary_pattern(blurred_image, lbp_points, lbp_radius, method='uniform')
    
    return lbp.flatten()

def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height))

def load_images_from_folders(positive_folder, negative_folder, new_width, new_height):
    def load_images_from_folder(folder):
        images = []
        image_paths = []
        for root, _, files in os.walk(folder):
            for filename in files:
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = resize_image(img, new_width, new_height)
                    images.append(img_resized)
                    image_paths.append(img_path)
        return images, image_paths

    positive_images, positive_image_paths = load_images_from_folder(positive_folder)
    negative_images, negative_image_paths = load_images_from_folder(negative_folder)

    return positive_images, negative_images, positive_image_paths, negative_image_paths

# Specify the paths to the positive and negative folders
positive_folder = r'traindataset\positive'
negative_folder = r'traindataset\negative'

# Set the new width and height for resizing
new_width = 320
new_height = 180

# Load resized images from positive and negative folders
positive_images, negative_images, positive_image_paths, negative_image_paths = load_images_from_folders(positive_folder, negative_folder, new_width, new_height)

# Check if there are images in both classes
if not positive_images or not negative_images:
    print("Error: Insufficient data. Ensure there are images in both positive and negative classes.")
    exit()

images = positive_images + negative_images
positive_labels = np.ones(len(positive_images))
negative_labels = np.zeros(len(negative_images))
labels = np.concatenate([positive_labels, negative_labels])
all_image_paths = positive_image_paths + negative_image_paths

# Extract LBP features for each image
lbp_features = [compute_lbp_features(img) for img in images]
if len(images) == 0 or len(labels) == 0:
    print("Error: Insufficient data. Ensure there are images in both positive and negative classes.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, train_image_paths, test_image_paths = train_test_split(lbp_features, labels, all_image_paths, test_size=0.2, random_state=42)

if len(X_train) == 0 or len(y_train) == 0:
    print("Error: Insufficient data. Ensure there are images in both positive and negative classes.")
    exit()

# Training --> the SVM classifier
svm_classifier_lbp = SVC(kernel='linear')
svm_classifier_lbp.fit(X_train, y_train)

# Save the trained classifier and change the name upon using a different image size other than in the train folder
joblib.dump(svm_classifier_lbp, 'trained_classifier_lbp.pkl')

# Evaluate the classifier on the test set
y_pred = svm_classifier_lbp.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
