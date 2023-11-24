import cv2
import mediapipe as mp
import time
import pyautogui

def move_arrow_key(direction):
    if direction == "right":
        pyautogui.press('right')
    elif direction == "left":
        pyautogui.press('left')
    elif direction == "up":
        pyautogui.press('up')
    elif direction == "down":
        pyautogui.press('down')

def detect_hand_movement():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)  # Limit to one hand for better performance

    mp_drawing = mp.solutions.drawing_utils

    prev_index_tip_x = 0
    prev_index_tip_y = 0

    last_movement_time = time.time()

    frame_skip_count = 0
    frame_skip_interval = 3  # Skip every 3 frames

    movement_threshold = 10  # Adjust this threshold based on your needs

    while True:
        ret, frame = cap.read()

        # if frame_skip_count < frame_skip_interval:
        #     frame_skip_count += 1
        #     continue
        # else:
        #     frame_skip_count = 0

        # Resize frame to reduce processing time
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_x, index_tip_y = int(index_tip_landmark.x * small_frame.shape[1]), int(index_tip_landmark.y * small_frame.shape[0])

                x_difference = index_tip_x - prev_index_tip_x
                y_difference = index_tip_y - prev_index_tip_y

                prev_index_tip_x, prev_index_tip_y = index_tip_x, index_tip_y

                if abs(x_difference) > movement_threshold or abs(y_difference) > movement_threshold:
                    if abs(x_difference) > abs(y_difference):
                        if x_difference > 0:
                            move_arrow_key("right")
                            print("Direction: Right")
                        elif x_difference < 0:
                            move_arrow_key("left")
                            print("Direction: Left")
                    else:
                        if y_difference > 0:
                            move_arrow_key("down")
                            print("Direction: Down")
                        elif y_difference < 0:
                            move_arrow_key("up")
                            print("Direction: Up")

                    last_movement_time = current_time

        if current_time - last_movement_time > 1.0:
            print("Finger not moving")

        # if results.multi_hand_landmarks:
        #     # Scale landmarks back to the original frame size for drawing
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x *= frame.shape[1]
        #         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y *= frame.shape[0]

        #     mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        #cv2.imshow("Hand Movement Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time hand detection and movement detection using MediaPipe
detect_hand_movement()
