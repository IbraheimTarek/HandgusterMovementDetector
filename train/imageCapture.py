import cv2
import os

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

cap = cv2.VideoCapture(0)

image_count = 0
positive_images = []
negative_images = []

while True:
    ret, frame = cap.read()
    cv2.imshow('Capture Images', frame)

    key = cv2.waitKey(1) & 0xFF

    # 'p' key for positive image
    if key == ord('p'):
        positive_images.append(frame)
        print(f'Captured positive image {len(positive_images)}')

    # 'n' key for negative image
    elif key == ord('n'):
        negative_images.append(frame)
        print(f'Captured negative image {len(negative_images)}')

    # 'q' key to quit
    elif key == ord('q'):
        break

# Release the camera
cap.release()

# Close the window
cv2.destroyAllWindows()

# Apply Gaussian blur to positive images and save
for i, img in enumerate(positive_images):
    blurred_img = apply_gaussian_blur(img)
    cv2.imwrite(f'positive_image_{i + 1}.jpg', blurred_img)

# Apply Gaussian blur to negative images and save
for i, img in enumerate(negative_images):
    blurred_img = apply_gaussian_blur(img)
    cv2.imwrite(f'negative_image_{i + 1}.jpg', blurred_img)

print(f'Total positive images: {len(positive_images)}')
print(f'Total negative images: {len(negative_images)}')
