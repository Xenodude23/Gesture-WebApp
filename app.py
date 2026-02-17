import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
import math
import time
import pyautogui
from ultralytics import YOLO
from collections import deque
import numpy as np
from pynput import keyboard
import threading

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_COMPLEXITY = 0
EYE_AR_THRESH = 0.25
PINCH_THRESHOLD = 0.05
DROWSINESS_TIME_THRESH = 2.0
OBJECT_DETECT_INTERVAL = 8

# Zoom Settings
ZOOM_HOLD_TIME = 1.0

# Stress Meter
STRESS_DECAY = 0.5
JITTER_THRESHOLD = 0.05

# --- INITIALIZE AI MODELS ---
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
object_model = YOLO('yolov8s.pt')

pyautogui.FAILSAFE = False

# --- DATA STORAGE ---
blink_timestamps = deque(maxlen=30)
prev_nose_pos = (0, 0)
last_description_time = 0
fps_history = deque(maxlen=60)
last_boxes = []

# Mouse Safety (Spacebar Toggle)
mouse_locked = True

# Zoom Variables
zoom_alpha = 0.0
is_zoom_target = False
peace_start_time = 0

# Orb Variables
orb_active = False  # Is the orb on screen?
orb_grabbed = False  # Are we dragging it?
orb_x, orb_y = 0, 0  # Current Orb position
orb_rotation = 0.0


# --- KEYBOARD LISTENER (Spacebar) ---
def on_press(key):
    global mouse_locked
    if key == keyboard.Key.space:
        mouse_locked = not mouse_locked
        print(f"Mouse Status: {'LOCKED' if mouse_locked else 'UNLOCKED'}")


# Start listener in a non-blocking thread
listener = keyboard.Listener(on_press=on_press)
listener.start()


# --- HELPER FUNCTIONS ---

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def lerp(a, b, t):
    return a + (b - a) * t


def get_eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    vertical_1 = calculate_distance(p2, p6)
    vertical_2 = calculate_distance(p3, p5)
    horizontal = calculate_distance(p1, p4)
    if horizontal == 0: return 0
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def is_fist(landmarks):
    wrist = landmarks[0]
    tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    avg_dist = 0
    for tip in tips:
        avg_dist += calculate_distance(tip, wrist)
    avg_dist /= 4
    return avg_dist < 0.15  # Low distance = Fist


def is_peace_sign(landmarks):
    """Index and Middle UP, Ring and Pinky DOWN"""
    wrist = landmarks[0]
    # Up fingers
    idx_tip = landmarks[8]
    mid_tip = landmarks[12]
    # Down fingers
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check if Ring/Pinky are close to wrist (curled)
    is_ring_curled = calculate_distance(ring_tip, wrist) < 0.15
    is_pinky_curled = calculate_distance(pinky_tip, wrist) < 0.15

    # Check if Index/Middle are far from wrist (extended)
    is_idx_extended = calculate_distance(idx_tip, wrist) > 0.2
    is_mid_extended = calculate_distance(mid_tip, wrist) > 0.2

    return is_idx_extended and is_mid_extended and is_ring_curled and is_pinky_curled


def apply_zoom(image, center_x, center_y, zoom_factor):
    h, w = image.shape[:2]
    if zoom_factor > 0.95: zoom_factor = 0.95
    crop_w = int(w * (1.0 - (0.7 * zoom_factor)))
    crop_h = int(h * (1.0 - (0.7 * zoom_factor)))
    x1 = max(0, int(center_x - crop_w // 2))
    y1 = max(0, int(center_y - crop_h // 2))
    x2 = min(w, int(center_x + crop_w // 2))
    y2 = min(h, int(center_y + crop_h // 2))
    zoomed_img = image[y1:y2, x1:x2]
    return cv2.resize(zoomed_img, (w, h))


def draw_3d_orb(img, center_x, center_y, radius, rotation):
    overlay = img.copy()
    # 1. Core
    cv2.circle(overlay, (center_x, center_y), int(radius / 4), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # 2. Rings
    num_rings = 3
    for i in range(num_rings):
        angle_offset = i * (math.pi / num_rings)
        current_rotation = rotation + angle_offset
        width = int(radius * math.cos(current_rotation))
        height = int(radius * 0.4)
        if width > 0:
            cv2.ellipse(img, (center_x, center_y), (abs(width), height),
                        0, 0, 360, (0, 255, 255), 1)
    # 3. Particles
    num_particles = 8
    for i in range(num_particles):
        p_angle = rotation + (i * (2 * math.pi / num_particles))
        p_radius = radius * 1.2
        px = int(center_x + p_radius * math.cos(p_angle))
        py = int(center_y + p_radius * math.sin(p_angle))
        cv2.circle(img, (px, py), 2, (255, 255, 0), -1)


def gen_frames():
    global last_boxes, prev_nose_pos, last_description_time, fps_history, zoom_alpha, is_zoom_target, peace_start_time, orb_active, orb_grabbed, orb_x, orb_y, orb_rotation

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    screen_w, screen_h = pyautogui.size()
    prev_x, prev_y = screen_w // 2, screen_h // 2
    is_clicking = False
    last_click_time = 0
    stress_level = 0.0

    with mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=MODEL_COMPLEXITY,
            refine_face_landmarks=False
    ) as holistic:

        prev_time = time.time()
        eye_closed_start_time = 0
        frame_count = 0

        WHITE = (255, 255, 255)
        YELLOW = (0, 255, 255)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 1. Pre-process
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_holistic = holistic.process(image_rgb)

            # 2. Object Detection
            curr_time = time.time()
            results_objects = None

            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            if frame_count % OBJECT_DETECT_INTERVAL == 0:
                results_objects = object_model(image_rgb, verbose=False)

            # 3. Prepare for Drawing
            image.flags.writeable = True
            h, w, c = image.shape
            frame_count += 1

            # --- GESTURE DETECTION ---
            current_fist = False
            current_peace = False
            current_pinch = False
            hand_detected = False
            hand_x, hand_y = 0, 0

            # Check Hands
            if results_holistic.left_hand_landmarks:
                hand_detected = True
                current_fist = is_fist(results_holistic.left_hand_landmarks.landmark)
                if not current_fist: current_peace = is_peace_sign(results_holistic.left_hand_landmarks.landmark)

                idx_lm = results_holistic.left_hand_landmarks.landmark[8]
                thumb_lm = results_holistic.left_hand_landmarks.landmark[4]
                hand_x, hand_y = int(idx_lm.x * w), int(idx_lm.y * h)
                if calculate_distance(idx_lm, thumb_lm) < PINCH_THRESHOLD: current_pinch = True

            elif results_holistic.right_hand_landmarks:
                hand_detected = True
                current_fist = is_fist(results_holistic.right_hand_landmarks.landmark)
                if not current_fist: current_peace = is_peace_sign(results_holistic.right_hand_landmarks.landmark)

                idx_lm = results_holistic.right_hand_landmarks.landmark[8]
                thumb_lm = results_holistic.right_hand_landmarks.landmark[4]
                hand_x, hand_y = int(idx_lm.x * w), int(idx_lm.y * h)
                if calculate_distance(idx_lm, thumb_lm) < PINCH_THRESHOLD: current_pinch = True

            # --- ORB STATE MACHINE ---
            if current_fist:
                if not orb_active:
                    # Spawn Orb
                    orb_active = True
                    orb_x, orb_y = hand_x, hand_y
                    orb_grabbed = False  # Start idle
                elif orb_active and not orb_grabbed:
                    # Despawn Orb (Only if we are grabbing nothing)
                    orb_active = False
            else:
                # Not fist
                pass

            if orb_active:
                orb_rotation += 0.1

                if current_pinch:
                    # Grab Orb
                    orb_grabbed = True
                    # Move Orb to Hand
                    orb_x, orb_y = hand_x, hand_y
                else:
                    # Release Orb (Stay where it is)
                    orb_grabbed = False

                draw_3d_orb(image, orb_x, orb_y, 60, orb_rotation)
                cv2.putText(image, "ORB ACTIVE", (orb_x + 40, orb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

            # --- ZOOM (Peace Sign) ---
            if current_peace:
                if peace_start_time == 0:
                    peace_start_time = curr_time
                if curr_time - peace_start_time > ZOOM_HOLD_TIME:
                    is_zoom_target = True
            else:
                peace_start_time = 0
                is_zoom_target = False

            target_alpha = 1.0 if is_zoom_target else 0.0
            zoom_alpha = lerp(zoom_alpha, target_alpha, 0.1)

            # --- OBJECT DETECTION ---
            if results_objects:
                last_boxes = []
                for box in results_objects[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    name = results_objects[0].names[cls_id]
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    if conf > 0.40:
                        if name != 'person':
                            last_boxes.append({'coords': (x1, y1, x2, y2), 'name': name})

            for box_data in last_boxes:
                x1, y1, x2, y2 = box_data['coords']
                name = box_data['name']
                cv2.rectangle(image, (x1, y1), (x2, y2), BLUE, 1)
                cv2.putText(image, name, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)

            # --- FACE ANALYSIS ---
            head_direction = "CENTER"
            head_angle = 0
            is_drowsy = False
            nose_pos = None
            face_center = (w // 2, h // 2)

            if results_holistic.face_landmarks:
                landmarks = results_holistic.face_landmarks.landmark
                try:
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]
                    left_ear = get_eye_aspect_ratio(landmarks, left_eye_indices)
                    right_ear = get_eye_aspect_ratio(landmarks, right_eye_indices)
                    avg_ear = (left_ear + right_ear) / 2.0

                    eye_status = "OPEN" if avg_ear > EYE_AR_THRESH else "CLOSED"
                    eye_color = GREEN if eye_status == "OPEN" else RED

                    if eye_status == "CLOSED":
                        if eye_closed_start_time == 0:
                            eye_closed_start_time = time.time()
                        elif time.time() - eye_closed_start_time > DROWSINESS_TIME_THRESH:
                            is_drowsy = True
                    else:
                        eye_closed_start_time = 0

                    nose = landmarks[1]
                    nose_pos = (nose.x, nose.y)
                    left_cheek = landmarks[234]
                    right_cheek = landmarks[454]

                    face_center_x = int((left_cheek.x + right_cheek.x) * w / 2)
                    face_center_y = int((left_cheek.y + right_cheek.y) * h / 2)
                    face_center = (face_center_x, face_center_y)

                    dist_left = calculate_distance(nose, left_cheek)
                    dist_right = calculate_distance(nose, right_cheek)
                    if dist_left > dist_right + 0.05:
                        head_direction = "RIGHT"
                    elif dist_right > dist_left + 0.05:
                        head_direction = "LEFT"

                    dy = right_cheek.y - left_cheek.y
                    dx = right_cheek.x - left_cheek.x
                    head_angle = math.atan2(dy, dx)

                    if prev_nose_pos != (0, 0):
                        jitter_dist = calculate_distance({'x': nose.x, 'y': nose.y},
                                                         {'x': prev_nose_pos[0], 'y': prev_nose_pos[1]})
                        if jitter_dist > JITTER_THRESHOLD: stress_level += 5
                    prev_nose_pos = (nose.x, nose.y)

                    cv2.putText(image, f"EYE: {eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, eye_color, 1)
                    cv2.putText(image, f"LOOK: {head_direction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)
                    if is_drowsy:
                        cv2.putText(image, "!!! DROWSY !!!", (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED,
                                    2)
                except:
                    pass

            if zoom_alpha > 0.05:
                image = apply_zoom(image, face_center[0], face_center[1], zoom_alpha)
                cv2.putText(image, "ZOOM: ACTIVE", (w - 130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

            # --- STRESS METER ---
            stress_level = max(0, stress_level - STRESS_DECAY)
            stress_color = GREEN if stress_level < 30 else (YELLOW if stress_level < 60 else RED)
            cv2.putText(image, f"STRESS: {int(stress_level)}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, stress_color,
                        1)
            bar_len = int((stress_level / 100) * 100)
            cv2.rectangle(image, (10, 85), (110, 95), (255, 255, 255), 1)
            cv2.rectangle(image, (10, 85), (10 + bar_len, 95), stress_color, -1)

            # --- MOUSE SAFETY UI ---
            lock_color = RED if mouse_locked else GREEN
            lock_text = "MOUSE LOCKED [SPACE]" if mouse_locked else "MOUSE UNLOCKED [SPACE]"
            cv2.putText(image, lock_text, (w - 220, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2)

            # --- HAND TRACKING & MOUSE ---
            if hand_detected:
                # Mouse Control (Spacebar check)
                if not mouse_locked:
                    target_x = int(screen_w * (1 - (hand_x / w)))
                    target_y = int(screen_h * (hand_y / h))

                    smooth_x = int(lerp(prev_x, target_x, 0.3))
                    smooth_y = int(lerp(prev_y, target_y, 0.3))

                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y

                    # Click Logic (Pinch) - Only if NOT grabbing Orb (to avoid conflict)
                    if current_pinch and not orb_grabbed:
                        if not is_clicking and (time.time() - last_click_time > 0.5):
                            pyautogui.click()
                            is_clicking = True
                            last_click_time = time.time()
                            cv2.putText(image, "CLICK", (w // 2 - 30, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        YELLOW, 2)
                    else:
                        is_clicking = False

                # Draw Hand Lines
                # We need to access hand_landmarks again for drawing if we didn't save the list earlier
                # For simplicity in this optimized loop, we draw generic lines or access from results_holistic directly
                if results_holistic.left_hand_landmarks:
                    # Quick draw function
                    for connection in mp_holistic.HAND_CONNECTIONS:
                        s = results_holistic.left_hand_landmarks.landmark[connection[0]]
                        e = results_holistic.left_hand_landmarks.landmark[connection[1]]
                        sx, sy = int(s.x * w), int(s.y * h)
                        ex, ey = int(e.x * w), int(e.y * h)
                        cv2.line(image, (sx, sy), (ex, ey), WHITE, 1)
                    for idx, lm in enumerate(results_holistic.left_hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        color = GREEN if idx in [4, 8, 12, 16, 20] else WHITE
                        cv2.rectangle(image, (cx - 2, cy - 2), (cx + 2, cy + 2), color, 1)

                elif results_holistic.right_hand_landmarks:
                    for connection in mp_holistic.HAND_CONNECTIONS:
                        s = results_holistic.right_hand_landmarks.landmark[connection[0]]
                        e = results_holistic.right_hand_landmarks.landmark[connection[1]]
                        sx, sy = int(s.x * w), int(s.y * h)
                        ex, ey = int(e.x * w), int(e.y * h)
                        cv2.line(image, (sx, sy), (ex, ey), WHITE, 1)
                    for idx, lm in enumerate(results_holistic.right_hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        color = GREEN if idx in [4, 8, 12, 16, 20] else WHITE
                        cv2.rectangle(image, (cx - 2, cy - 2), (cx + 2, cy + 2), color, 1)

            cv2.putText(image, f"FPS: {int(avg_fps)}", (w - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feedback')
def video_feedback():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
