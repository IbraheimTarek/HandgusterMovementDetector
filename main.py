import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.filters import gaussian
from skimage import exposure
from scipy.stats import mode
from argparse import ArgumentParser
import joblib
import time
import pyautogui
import threading
import queue 


def compute_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred_image = gaussian(gray, sigma=0.01)

    # Apply histogram equalization for contrast enhancement
    #enhanced_image = exposure.equalize_hist(blurred_image)
    #cv2.imshow('Original Frame gray', blurred_image)
    # Compute HOG features
    hog_features, hog_image = hog(blurred_image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

    return hog_features,hog_image



def segment_hand_by_color(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV color range for skin color (adjust these values as needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 200, 200], dtype=np.uint8)

    # Create a binary mask for the skin color range
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Bitwise AND operation to get the segmented hand region
    segmented_hand = cv2.bitwise_and(image, image, mask=skin_mask)

    return segmented_hand




def process_hog_features(frame_queue):
    while not terminate_signal.is_set():
        try:
            if not frame_queue.empty():
                # Get a frame from the queue
                frame = frame_queue.get()

                # Compute HOG features
                hog_features,hog_image = compute_hog_features(frame)
                orientation_prediction = predict_class_hog(hog_features,svm_classifier)
                print(bool(orientation_prediction))
                if(bool(orientation_prediction)) :
                    args['classifier_res'] = bool(orientation_prediction)
                    time.sleep(0.5)

                #print(frame_queue.qsize())
                
                # Display the HOG image in real-time
                cv2.imshow('HOG Image', hog_image)
                frame_queue.task_done()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except queue.Empty:
            pass  # Ignore empty queue (no new frames)


def output_direction(loc):
    while not terminate_signal.is_set():
        classifier_output = args['classifier_res']
        if loc == 0 and classifier_output:
            pyautogui.press('down')
        elif loc == 1 and classifier_output:
            pyautogui.press('right')
        elif loc == 2 and classifier_output:
            pyautogui.press('up')
        elif loc == 3 and classifier_output:
            pyautogui.press('left')
        else:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def predict_class_hog(hog_features, classifier):
    # Ensure hog_features is a 2D array
    hog_features = hog_features.reshape(1, -1)

    # Predict the class
    predicted_class = classifier.predict(hog_features)
    return predicted_class[0]


max_queue_size = 10
frame_queue = queue.Queue(maxsize=max_queue_size)

svm_classifier = joblib.load('trained_classifier_V9.pkl')
# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)
terminate_signal = threading.Event()
# Set the desired frame width and height
frame_width = 320
frame_height = 180

# Set the frame width and height for the camera
cap.set(3, frame_width)
cap.set(4, frame_height)

ap = ArgumentParser()
ap.add_argument('-p', '--plot', default=False, action='store_true', help='Plot accumulators?')
ap.add_argument('-rgb', '--rgb', default=False, action='store_true', help='Show RGB mask?')
ap.add_argument('-classifier_res', '--classifier_res', default=False, action='store_true', help='Show classifier output?')
args = vars(ap.parse_args())

directions_map = np.zeros([10, 5])

if args['plot']:
    plt.ion()

param = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 8,#15
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}


ret, gray_previous = cap.read()

hsv = np.zeros_like(gray_previous)
hsv[:, :, 1] = 255

gray_previous = cv2.cvtColor(gray_previous, cv2.COLOR_BGR2GRAY)

loc = 0 
hog_thread = threading.Thread(target=process_hog_features, args=(frame_queue,))
hog_thread.start()
#pyautogui_thread = threading.Thread(target=output_direction, args=(loc,))
#pyautogui_thread.start()
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # # Compute the orientation difference map
    # result_map_orientation = cv2.Canny(frame, threshold, threshold * 2)

    # # Compute color difference mask
    # color_mask = segment_hand_by_color(frame)

    # result_map_orientation = cv2.cvtColor(result_map_orientation, cv2.COLOR_GRAY2BGR)

    # merged_map = cv2.addWeighted(color_mask, 0.5, result_map_orientation, 1, 0)
    # cv2.imshow('canny Frame', result_map_orientation)
    # cv2.imshow('colormask Frame', color_mask)
    # cv2.imshow('combined Frame', merged_map)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray_previous, gray, None, **param)

    # Calculate magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees=True)

    # Threshold for motion detection
    threshold = 2.0

    # Create a binary mask based on the magnitude threshold
    mask = np.zeros_like(gray)
    mask[magnitude > threshold] = 255

    # Apply the mask to the original frame
    moving_objects = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Moving Objects', moving_objects)

    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray_blurred_image = gaussian(frame_gray, sigma=0.1)
    # moving_objects_gray = cv2.cvtColor(moving_objects, cv2.COLOR_BGR2GRAY)
    # moving_objects_gray_blurred_image = gaussian(moving_objects_gray, sigma=1)
    #cv2.imshow('Original Frame gray', frame_gray_blurred_image)
    # cv2.imshow('Moving Objects gray', moving_objects_gray_blurred_image)
    try:
        # Put the frame in the queue
        frame_queue.put_nowait(moving_objects)
    except queue.Full:
        # Handle the queue full situation (e.g., remove the oldest item)
        oldest_frame = frame_queue.get_nowait()
        frame_queue.put_nowait(moving_objects)



    ang_180 = angle/2
    gray_previous = gray
    
    move_sense = angle[magnitude > 10 ]
    move_mode = mode(move_sense)[0] # module to calculate the mode (most frequent value) of the array

    if 10 < move_mode <= 100:
        directions_map[-1, 0] = 1
        directions_map[-1, 1:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 100 < move_mode <= 190:
        directions_map[-1, 1] = 1
        directions_map[-1, :1] = 0
        directions_map[-1, 2:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 190 < move_mode <= 280:
        directions_map[-1, 2] = 1
        directions_map[-1, :2] = 0
        directions_map[-1, 3:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 280 < move_mode or move_mode < 10:
        directions_map[-1, 3] = 1
        directions_map[-1, :3] = 0
        directions_map[-1, 4:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    else:
        directions_map[-1, -1] = 1
        directions_map[-1, :-1] = 0
        directions_map = np.roll(directions_map, 1, axis=0)

    if args['plot']:
        plt.clf()
        plt.plot(directions_map[:, 0], label='Down')
        plt.plot(directions_map[:, 1], label='Right')
        plt.plot(directions_map[:, 2], label='Up')
        plt.plot(directions_map[:, 3], label='Left')
        plt.plot(directions_map[:, 4], label='Waiting')
        plt.legend(loc=2)
        plt.pause(1e-5)
        plt.show()

    loc = directions_map.mean(axis=0).argmax()
    classifier_output = args['classifier_res']
    if loc == 0 and classifier_output:
        text = 'Moving down'
    elif loc == 1 and classifier_output:
        text = 'Moving right'
    elif loc == 2 and classifier_output:
        text = 'Moving up'
    elif loc == 3 and classifier_output:
        text = 'Moving left'
    else:
        text = 'WAITING'

    hsv[:, :, 0] = ang_180
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, text, (30, 90), cv2.FONT_HERSHEY_TRIPLEX, frame.shape[1] / 500, (0, 0, 255), 2)

    if args['rgb']:
        cv2.imshow('Mask', rgb)
    cv2.imshow('Frame', frame)


    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        terminate_signal.set()
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
