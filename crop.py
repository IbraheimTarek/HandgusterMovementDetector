import cv2
import numpy as np

# Read the image
image = cv2.imread('train/positive_image_16.jpg')

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the skin color in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Threshold the image to get the binary mask for the skin color
skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

# Find contours in the skin mask
contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the bounding box of the largest contour (assuming it's the hand)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the hand region
    hand_region = image[y:y + h, x:x + w]

    # Draw contours on the original image
    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)

    # Display the original and segmented images
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Hand', hand_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No hand detected in the image.")
