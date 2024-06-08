import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import face_recognition
import os
import time

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Set the screen resolution of your laptop
screen_width = 1920
screen_height = 1080

# Smoothing factor for cursor movement
smoothing = 7  # Increase the smoothing factor for smoother movement
prev_x, prev_y = 0, 0

def detect_gesture(landmarks):
    """
    Custom logic to detect gestures.
    Returns the gesture detected.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Pinch gesture for click
    if abs(index_tip.x - thumb_tip.x) < 0.03 and abs(index_tip.y - thumb_tip.y) < 0.03:
        return "click"
    
    # "Two" gesture for volume up
    if (index_tip.y < landmarks[7].y and  # Index finger is up
        middle_tip.y < landmarks[11].y and  # Middle finger is up
        ring_tip.y > landmarks[15].y and  # Ring finger is down
        pinky_tip.y > landmarks[19].y and  # Pinky finger is down
        thumb_tip.y > landmarks[3].y):  # Thumb is down
        return "volume_up"
    
    # Fist gesture for volume down
    if (index_tip.y > landmarks[7].y and  # Index finger is down
        middle_tip.y > landmarks[11].y and  # Middle finger is down
        ring_tip.y > landmarks[15].y and  # Ring finger is down
        pinky_tip.y > landmarks[19].y and  # Pinky finger is down
        thumb_tip.y > landmarks[3].y):  # Thumb is down
        return "volume_down"
    
    return None

# Load the image of the authorized user and encode the face
authorized_image_path = "../assets/authorized_user.jpg"
if not os.path.exists(authorized_image_path):
    print("Authorized user image not found. Please place 'authorized_user.jpg' in the 'assets' directory.")
    cap.release()
    exit()

authorized_image = face_recognition.load_image_file(authorized_image_path)
authorized_encoding = face_recognition.face_encodings(authorized_image)[0]

face_recognized = False

# Target FPS
target_fps = 30
prev_frame_time = 0
frame_delay = 1 / target_fps

while True:
    current_time = time.time()
    elapsed_time = current_time - prev_frame_time

    if elapsed_time >= frame_delay:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not face_recognized:
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([authorized_encoding], face_encoding)
                if True in matches:
                    face_recognized = True
                    # print("Face recognized! Virtual mouse unlocked.")
                    break

        if face_recognized:
            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Detect gesture
                    gesture = detect_gesture(hand_landmarks.landmark)
                    
                    if gesture == "click":
                        pyautogui.click()
                    elif gesture == "volume_up":
                        pyautogui.press("volumeup")
                    elif gesture == "volume_down":
                        pyautogui.press("volumedown")

                    # Get the coordinates of the index finger tip
                    index_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])

                    # Convert the coordinates to screen space
                    screen_x = np.interp(cx, (0, img.shape[1]), (0, screen_width))
                    screen_y = np.interp(cy, (0, img.shape[0]), (0, screen_height))

                    # Smooth the cursor movement
                    cur_x = prev_x + (screen_x - prev_x) / smoothing
                    cur_y = prev_y + (screen_y - prev_y) / smoothing

                    # Move the cursor
                    pyautogui.moveTo(screen_width - cur_x, cur_y)
                    prev_x, prev_y = cur_x, cur_y

            cv2.imshow("Hand Tracking", img)
        
        prev_frame_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
