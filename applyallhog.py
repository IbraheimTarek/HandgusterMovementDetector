import cv2
from skimage.feature import hog
from skimage.filters import gaussian
import os
from matplotlib import pyplot as plt

def compute_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = gaussian(gray, sigma=0.01)
    hog_features, hog_image = hog(blurred_image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

    return hog_features, hog_image

def process_images(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(folder_path, image_file)
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (320, 180))
        # Compute HOG features
        hog_features, hog_image = compute_hog_features(original_image)

        # Display the original and HOG-processed images side by side
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Processed Image')
        plt.axis('off')

        plt.show()

if __name__ == "__main__":
    folder_path = "C:/Users/HIMA/source/repos/IPProject/traindataset/positive/01_palm"
    process_images(folder_path)
