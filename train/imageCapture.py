import cv2
import numpy as np

cap = cv2.VideoCapture(0)

image_count = 0
positive_images = []
negative_images = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (32, 32))
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

cap.release()
cv2.destroyAllWindows()

for i, img in enumerate(positive_images):
    cv2.imwrite(f'positive_image_{i+1}.jpg', img)

for i, img in enumerate(negative_images):
    cv2.imwrite(f'negative_image_{i+1}.jpg', img)

print(f'Total positive images: {len(positive_images)}')
print(f'Total negative images: {len(negative_images)}')
