import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Calculate face angle
def get_face_angle(landmarks, image_width, image_height, frame):
    # Get required landmarks
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Get pixel positions
    nose_x = int(nose_tip.x * image_width)
    left_x = int(left_eye.x * image_width)
    right_x = int(right_eye.x * image_width)

    # Compute midpoint between eyes
    mid_x = (left_x + right_x) // 2
    face_width = abs(right_x - left_x)

    # Compute normalized offset
    offset = (nose_x - mid_x) / face_width  # e.g., 0.0 = center, >0 = right, <0 = left

    # Use percentage threshold (e.g., 10% of face width)
    threshold = 0.10  # 8% of face width

    # Determine face position
    if offset > threshold:
        pos = "Right"
    elif offset < -threshold:
        pos = "Left"
    else:
        pos = "Center"

    # Draw guidance lines
    cv2.line(frame, (left_x, int(nose_tip.y * image_height)), (right_x, int(nose_tip.y * image_height)), (200, 200, 200), 1)
    cv2.line(frame, (nose_x, int(nose_tip.y * image_height) - 20), (nose_x, int(nose_tip.y * image_height) + 20), (0, 255, 0), 2)

    # Display result
    cv2.putText(frame, f"Face: {pos} ({offset:.2f})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return offset, pos

# Detect thumbs up/down
def detect_thumb_direction(hand_landmarks, image_width, image_height, frame):
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]

    wrist_pt = (int(wrist.x * image_width), int(wrist.y * image_height))
    thumb_pt = (int(thumb_tip.x * image_width), int(thumb_tip.y * image_height))

    cv2.line(frame, wrist_pt, thumb_pt, (255, 0, 0), 3)

    # Heuristic: thumb higher than wrist and index knuckle folded
    is_up = thumb_tip.y < wrist.y and index_mcp.y > thumb_tip.y
    is_down = thumb_tip.y > wrist.y and index_mcp.y < thumb_tip.y

    if is_up:
        label = "Thumbs Up"
        color = (0, 255, 0)
    elif is_down:
        label = "Thumbs Down"
        color = (0, 0, 255)
    else:
        return None

    cv2.putText(frame, label, (thumb_pt[0], thumb_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return label

# Init cam and models
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for selfie view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Process face first
    results_face = face_mesh.process(frame_rgb)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_FACE_OVAL,  # Only oval
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            get_face_angle(face_landmarks.landmark, image_width, image_height, frame)

    # Process hands
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_thumb_direction(hand_landmarks, image_width, image_height, frame)

    # Display
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
