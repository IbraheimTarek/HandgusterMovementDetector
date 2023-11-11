import cv2
import mediapipe as mp

def detect_hand_movement():
    '''Function to detect hand using MediaPipe'''
    cap = cv2.VideoCapture(0)
    # initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    # mediaPipe Drawing module Baby
    mp_drawing = mp.solutions.drawing_utils
    # previous coordinates of a specific landmark (e.g., base of the thumb)
    prev_x_coordinate = 0
    prev_y_coordinate = 0
    
    while True:
        ret, frame = cap.read()

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                x_coordinate = int(landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * frame.shape[1])
                y_coordinate = int(landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * frame.shape[0])

                x_difference = x_coordinate - prev_x_coordinate
                y_difference = y_coordinate - prev_y_coordinate

                prev_x_coordinate = x_coordinate
                prev_y_coordinate = y_coordinate

                if abs(x_difference) > abs(y_difference):
                    if x_difference > 0:
                        print("Hand moving right")
                    elif x_difference < 0:
                        print("Hand moving left")
                else:
                    if y_difference > 0:
                        print("Hand moving down")
                    elif y_difference < 0:
                        print("Hand moving up")

        cv2.imshow("Hand Movement Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time hand detection and movement detection using MediaPipe
detect_hand_movement()