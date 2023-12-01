import cv2
import os

cap = cv2.VideoCapture(0)

image_count = 0
positive_images = []
negative_images = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (64, 64))
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

# Save positive images
for i, img in enumerate(positive_images):
    # Find a unique filename
    img_number = 1
    while os.path.exists(f'positive_image_{img_number}.jpg'):
        img_number += 1

    cv2.imwrite(f'positive_image_{img_number}.jpg', img)

# Save negative images
for i, img in enumerate(negative_images):
    # Find a unique filename
    img_number = 1
    while os.path.exists(f'negative_image_{img_number}.jpg'):
        img_number += 1

    cv2.imwrite(f'negative_image_{img_number}.jpg', img)

print(f'Total positive images: {len(positive_images)}')
print(f'Total negative images: {len(negative_images)}')
