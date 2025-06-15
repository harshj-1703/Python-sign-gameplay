import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller, Key

keyboard_controller = Controller()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Timing
last_face_action_time = 0
cooldown_seconds = 0.1

# Thumb tracking
last_thumb_label = None
last_thumb_label_time = 0

# Key holding status
key_held = {
    'up': False,
    'down': False
}

def press_key_once(key):
    key_map = {'left': Key.left, 'right': Key.right}
    k = key_map.get(key.lower())
    if k:
        keyboard_controller.press(k)
        time.sleep(0.4)
        keyboard_controller.release(k)

def is_fist(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    folded_fingers = 0
    for tip_id in tip_ids[1:]:
        if hand_landmarks.landmark[tip_id].y > hand_landmarks.landmark[tip_id - 2].y:
            folded_fingers += 1
    return folded_fingers >= 4

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
    offset = (nose_x - mid_x) / face_width
    threshold = 0.10

    pos = "Center"
    if offset > threshold:
        pos = "Right"
    elif offset < -threshold:
        pos = "Left"

    current_time = time.time()
    if current_time - last_face_action_time > cooldown_seconds:
        if pos == "Left":
            print("← Pressed: Face Left")
            press_key_once('left')
        elif pos == "Right":
            print("→ Pressed: Face Right")
            press_key_once('right')
        last_face_action_time = current_time

    # Draw face lines
    cv2.line(frame, (left_x, int(nose_tip.y * image_height)),
             (right_x, int(nose_tip.y * image_height)), (200, 200, 200), 1)
    cv2.line(frame, (nose_x, int(nose_tip.y * image_height) - 20),
             (nose_x, int(nose_tip.y * image_height) + 20), (0, 255, 0), 2)
    cv2.putText(frame, f"Face: {pos} ({offset:.2f})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def detect_thumb_direction(hand_landmarks, image_width, image_height, frame, is_braking):
    global last_thumb_label, last_thumb_label_time, key_held

    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]

    wrist_pt = (int(wrist.x * image_width), int(wrist.y * image_height))
    thumb_pt = (int(thumb_tip.x * image_width), int(thumb_tip.y * image_height))
    cv2.line(frame, wrist_pt, thumb_pt, (255, 0, 0), 3)

    is_up = thumb_tip.y < wrist.y and index_mcp.y > thumb_tip.y
    is_down = thumb_tip.y > wrist.y and index_mcp.y < thumb_tip.y
    current_time = time.time()

    if is_braking:
        # Brake overrides everything
        if not key_held['down']:
            print("🛑 Brake: Both Hands Fist")
            keyboard_controller.press(Key.down)
            key_held['down'] = True
        if key_held['up']:
            keyboard_controller.release(Key.up)
            key_held['up'] = False
        last_thumb_label = "Brake (Both Fist)"
        last_thumb_label_time = current_time
        return

    # ACCELERATE
    if is_up:
        if not key_held['up']:
            print("↑ Holding: Thumbs Up")
            keyboard_controller.press(Key.up)
            key_held['up'] = True
        last_thumb_label = "Thumbs Up"
        last_thumb_label_time = current_time
    else:
        if key_held['up']:
            keyboard_controller.release(Key.up)
            key_held['up'] = False

    # REVERSE
    if is_down:
        if not key_held['down']:
            print("↓ Holding: Thumbs Down")
            keyboard_controller.press(Key.down)
            key_held['down'] = True
        last_thumb_label = "Thumbs Down"
        last_thumb_label_time = current_time
    else:
        if key_held['down']:
            keyboard_controller.release(Key.down)
            key_held['down'] = False

    # Show label
    if last_thumb_label and current_time - last_thumb_label_time <= 0.5:
        color = (0, 255, 0) if "Up" in last_thumb_label else (0, 0, 255)
        if "Brake" in last_thumb_label:
            color = (0, 255, 255)
        cv2.putText(frame, f"Gesture: {last_thumb_label}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Capture setup
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

    # Face control
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

    # Hand control
    results_hands = hands.process(frame_rgb)
    hands_detected = results_hands.multi_hand_landmarks

    is_braking = False
    if hands_detected and len(hands_detected) == 2:
        is_braking = is_fist(hands_detected[0]) and is_fist(hands_detected[1])
    elif key_held['down'] and last_thumb_label == "Brake (Both Fist)":
        keyboard_controller.release(Key.down)
        key_held['down'] = False

    if hands_detected:
        for hand_landmarks in hands_detected:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_thumb_direction(hand_landmarks, image_width, image_height, frame, is_braking)

    # Show frame
    cv2.imshow("Gesture Racing Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
