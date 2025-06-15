import cv2
import mediapipe as mp
import numpy as np
import math
import keyboard
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Timing to avoid multiple rapid inputs
last_face_action_time = 0
last_thumb_action_time = 0
cooldown_seconds = 1  # Minimum seconds between key presses

# Calculate face angle and send key press
def get_face_angle(landmarks, image_width, image_height, frame):
    global last_face_action_time

    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    nose_x = int(nose_tip.x * image_width)
    left_x = int(left_eye.x * image_width)
    right_x = int(right_eye.x * image_width)

    mid_x = (left_x + right_x) // 2
    face_width = abs(right_x - left_x)
    offset = (nose_x - mid_x) / face_width  # normalized offset
    threshold = 0.10  # Sensitivity

    # Determine position
    if offset > threshold:
        pos = "Right"
    elif offset < -threshold:
        pos = "Left"
    else:
        pos = "Center"

    # Key press logic with cooldown
    current_time = time.time()
    if current_time - last_face_action_time > cooldown_seconds:
        if pos == "Left":
            print("← Pressed: Face Left")
            keyboard.press_and_release('left')
            last_face_action_time = current_time
        elif pos == "Right":
            print("→ Pressed: Face Right")
            keyboard.press_and_release('right')
            last_face_action_time = current_time
        elif pos == "Center":
            print("Face Center (no key)")  # ❌ DO NOT reset last_face_action_time

    # Drawing
    cv2.line(frame, (left_x, int(nose_tip.y * image_height)), (right_x, int(nose_tip.y * image_height)), (200, 200, 200), 1)
    cv2.line(frame, (nose_x, int(nose_tip.y * image_height) - 20), (nose_x, int(nose_tip.y * image_height) + 20), (0, 255, 0), 2)

    cv2.putText(frame, f"Face: {pos} ({offset:.2f})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return offset, pos

# Detect thumbs up/down and send key press
def detect_thumb_direction(hand_landmarks, image_width, image_height, frame):
    global last_thumb_action_time, last_thumb_label, last_thumb_label_time

    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]

    wrist_pt = (int(wrist.x * image_width), int(wrist.y * image_height))
    thumb_pt = (int(thumb_tip.x * image_width), int(thumb_tip.y * image_height))

    cv2.line(frame, wrist_pt, thumb_pt, (255, 0, 0), 3)

    is_up = thumb_tip.y < wrist.y and index_mcp.y > thumb_tip.y
    is_down = thumb_tip.y > wrist.y and index_mcp.y < thumb_tip.y

    current_time = time.time()
    label = None
    color = (255, 255, 255)

    if is_up and current_time - last_thumb_action_time > cooldown_seconds:
        label = "Thumbs Up"
        color = (0, 255, 0)
        print("↑ Pressed: Thumbs Up")
        keyboard.press_and_release('up')
        last_thumb_action_time = current_time
        last_thumb_label = label
        last_thumb_label_time = current_time

    elif is_down and current_time - last_thumb_action_time > cooldown_seconds:
        label = "Thumbs Down"
        color = (0, 0, 255)
        print("↓ Pressed: Thumbs Down")
        keyboard.press_and_release('down')
        last_thumb_action_time = current_time
        last_thumb_label = label
        last_thumb_label_time = current_time

    # Always show last label for a short period
    if last_thumb_label and current_time - last_thumb_label_time <= 1.5:
        color = (0, 255, 0) if last_thumb_label == "Thumbs Up" else (0, 0, 255)
        cv2.putText(frame, f"Gesture: {last_thumb_label}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Init camera and models
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Face detection
    results_face = face_mesh.process(frame_rgb)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            get_face_angle(face_landmarks.landmark, image_width, image_height, frame)

    # Hand detection
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_thumb_direction(hand_landmarks, image_width, image_height, frame)

    cv2.imshow("Gesture Racing Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
