import cv2
import mediapipe as mp
import numpy as np
import time

# --- CONFIGURATION ---
WINDOW_NAME = "DLD FYP: Dual-Mode Communication System"
WIDTH, HEIGHT = 1280, 720

# Time in seconds the eye must be held closed to trigger command
EYE_HOLD_THRESHOLD = 1.0  # you specified 1 second

# Cooldowns (seconds) after a successful trigger to prevent bounce
EYE_TRIGGER_COOLDOWN = 1.5
GESTURE_TRIGGER_COOLDOWN = 1.0

# Automatic mode switch hold time (seconds)
AUTO_SWITCH_HOLD = 1.0

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.7  # Lowered detection conf for reliability
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- SYSTEM STATE ---
state = {
    "mode": "GESTURE",  # "GESTURE" or "BLINK"
    "light1": False,    # Controlled by left-eye hold or gestures
    "light2": False,    # Controlled by right-eye hold or gestures
    "fan": False,       # Controlled by both-eyes hold or gestures
    "last_action": "System Ready"
}

# --- BLINK DETECTION STATE (separate for left/right/both) ---
blink_state = {
    "left_closed": False,
    "right_closed": False,
    "both_closed": False,
    "left_start": 0.0,
    "right_start": 0.0,
    "both_start": 0.0,
    "left_cooldown": 0.0,
    "right_cooldown": 0.0,
    "both_cooldown": 0.0
}

# --- GESTURE STATE ---
gesture_state = {
    "last_gesture_time": 0.0
}

# --- AUTO MODE SWITCH STATE ---
auto_state = {
    "target_mode": None,
    "target_start": 0.0
}

# Eye landmarks (corrected for proper vertical centers; added horizontal for normalization)
LEFT_EYE_VERT = [159, 145]  # top, bottom (standard MediaPipe indices)
LEFT_EYE_HORIZ = [33, 133]  # outer, inner
RIGHT_EYE_VERT = [386, 374]  # top, bottom
RIGHT_EYE_HORIZ = [362, 263]  # outer, inner
EAR_THRESHOLD = 0.25    # Tuned threshold (common range 0.2-0.3; adjust based on prints)

def calculate_ear(landmarks, vert_indices, horiz_indices, h, w):
    """Calculate proper Eye Aspect Ratio (EAR) = vertical / horizontal."""
    if not landmarks:
        return 1.0
    
    # Vertical points
    top = np.array([landmarks[vert_indices[0]].x * w, landmarks[vert_indices[0]].y * h])
    bot = np.array([landmarks[vert_indices[1]].x * w, landmarks[vert_indices[1]].y * h])
    vertical = np.linalg.norm(top - bot)
    
    # Horizontal points
    left_pt = np.array([landmarks[horiz_indices[0]].x * w, landmarks[horiz_indices[0]].y * h])
    right_pt = np.array([landmarks[horiz_indices[1]].x * w, landmarks[horiz_indices[1]].y * h])
    horizontal = np.linalg.norm(left_pt - right_pt)
    
    return vertical / horizontal if horizontal > 0 else 0.0

def process_eye_holds(face_landmarks, h, w):
    """
    Update blink_state and return which trigger (if any) happened:
    returns one of: "LEFT", "RIGHT", "BOTH", or None
    """
    if not face_landmarks:
        # reset open states but keep cooldowns
        blink_state["left_closed"] = False
        blink_state["right_closed"] = False
        blink_state["both_closed"] = False
        blink_state["left_start"] = 0.0
        blink_state["right_start"] = 0.0
        blink_state["both_start"] = 0.0
        print("No face detected")  # Debug
        return None

    l_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_VERT, LEFT_EYE_HORIZ, h, w)
    r_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_VERT, RIGHT_EYE_HORIZ, h, w)
    print(f"Left EAR: {l_ear:.2f}, Right EAR: {r_ear:.2f}")  # Debug: monitor values (open ~0.3+, closed ~0.1-)
    
    l_closed = l_ear < EAR_THRESHOLD
    r_closed = r_ear < EAR_THRESHOLD

    now = time.time()
    triggered = None

    # Both eyes closed
    if l_closed and r_closed and now > blink_state["both_cooldown"]:
        if not blink_state["both_closed"]:
            blink_state["both_closed"] = True
            blink_state["both_start"] = now
        elapsed = now - blink_state["both_start"]
        if elapsed >= EYE_HOLD_THRESHOLD:
            triggered = "BOTH"
            blink_state["both_closed"] = False
            blink_state["both_cooldown"] = now + EYE_TRIGGER_COOLDOWN

            # also reset single eye trackers to avoid double triggers
            blink_state["left_closed"] = False
            blink_state["left_start"] = 0.0
            blink_state["left_cooldown"] = now + EYE_TRIGGER_COOLDOWN
            blink_state["right_closed"] = False
            blink_state["right_start"] = 0.0
            blink_state["right_cooldown"] = now + EYE_TRIGGER_COOLDOWN
        return triggered  # Early return to prioritize both

    # Left eye only
    if l_closed and not r_closed and now > blink_state["left_cooldown"]:
        if not blink_state["left_closed"]:
            blink_state["left_closed"] = True
            blink_state["left_start"] = now
        elapsed = now - blink_state["left_start"]
        if elapsed >= EYE_HOLD_THRESHOLD:
            triggered = "LEFT"
            blink_state["left_closed"] = False
            blink_state["left_cooldown"] = now + EYE_TRIGGER_COOLDOWN
            # prevent immediate both/right triggers
            blink_state["right_closed"] = False
            blink_state["right_start"] = 0.0
            blink_state["right_cooldown"] = now + 0.2
        return triggered

    # Right eye only
    if r_closed and not l_closed and now > blink_state["right_cooldown"]:
        if not blink_state["right_closed"]:
            blink_state["right_closed"] = True
            blink_state["right_start"] = now
        elapsed = now - blink_state["right_start"]
        if elapsed >= EYE_HOLD_THRESHOLD:
            triggered = "RIGHT"
            blink_state["right_closed"] = False
            blink_state["right_cooldown"] = now + EYE_TRIGGER_COOLDOWN
            # prevent immediate left trigger
            blink_state["left_closed"] = False
            blink_state["left_start"] = 0.0
            blink_state["left_cooldown"] = now + 0.2
        return triggered

    # If eyes are open, reset single timers (but not cooldown timers)
    if not l_closed:
        blink_state["left_closed"] = False
        blink_state["left_start"] = 0.0
    if not r_closed:
        blink_state["right_closed"] = False
        blink_state["right_start"] = 0.0
    # If both are not closed, reset both timer
    if not (l_closed and r_closed):
        blink_state["both_closed"] = False
        blink_state["both_start"] = 0.0

    return None

def count_fingers(hand_landmarks):
    """Counts extended fingers. Returns integer number of extended fingers (0-5)."""
    if not hand_landmarks:
        return 0
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: compare x for right hand mirrored feed (we flipped image)
    try:
        if lm[4].x < lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    except:
        fingers.append(0)

    # 4 fingers: tip (id) < pip (id-2) in y => extended
    for tip in [8, 12, 16, 20]:
        try:
            if lm[tip].y < lm[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        except:
            fingers.append(0)

    return sum(fingers)

def perform_gesture_action(finger_count):
    """Map finger_count to actions as requested, with cooldown checks."""
    now = time.time()
    if now < gesture_state["last_gesture_time"] + GESTURE_TRIGGER_COOLDOWN:
        return None  # still in cooldown

    # Map gestures
    if finger_count == 5:
        # Open palm = Turn ALL devices OFF
        state["light1"] = False
        state["light2"] = False
        state["fan"] = False
        state["last_action"] = "Open Palm -> ALL OFF"
    elif finger_count == 0:
        # Fist = Turn ALL devices ON
        state["light1"] = True
        state["light2"] = True
        state["fan"] = True
        state["last_action"] = "Fist -> ALL ON"
    elif finger_count == 2:
        # Two-finger = Toggle Light 1
        state["light1"] = not state["light1"]
        state["last_action"] = f"Two-Finger -> Light1 {'ON' if state['light1'] else 'OFF'}"
    elif finger_count == 4:
        # Four-finger = Toggle Fan
        state["fan"] = not state["fan"]
        state["last_action"] = f"Four-Finger -> Fan {'ON' if state['fan'] else 'OFF'}"
    elif finger_count == 1:
        # One finger = Toggle Light 2
        state["light2"] = not state["light2"]
        state["last_action"] = f"One-Finger -> Light2 {'ON' if state['light2'] else 'OFF'}"
    else:
        # other counts => do nothing
        return None

    gesture_state["last_gesture_time"] = now
    return state["last_action"]

def draw_dashboard(img, state, blink_state):
    """Dashboard with left/right/both progress bars and device status."""
    h, w, c = img.shape
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 360, 0), (w, h), (30, 30, 30), -1)  # Sidebar
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)  # Header
    alpha = 0.8
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Header
    cv2.putText(img, "FYP: BLINK & GESTURE SYSTEM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    x_base = w - 320
    # Mode
    cv2.putText(img, "SYSTEM MODE", (x_base, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    mode_color = (0, 255, 0) if state["mode"] == "GESTURE" else (0, 165, 255)
    cv2.rectangle(img, (x_base, 105), (w - 30, 155), mode_color, 2)
    cv2.putText(img, state["mode"], (x_base + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)

    # Blink hold indicators (Left / Right / Both)
    cv2.putText(img, "EYE-HOLD PROGRESS", (x_base, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    now = time.time()

    # Left eye progress
    left_elapsed = 0.0
    if blink_state["left_start"] > 0.0:
        left_elapsed = now - blink_state["left_start"]
    left_prog = min(1.0, left_elapsed / max(0.0001, EYE_HOLD_THRESHOLD))
    cv2.putText(img, "Left Eye", (x_base, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(img, (x_base, 215), (w - 30, 235), (50, 50, 50), -1)
    cv2.rectangle(img, (x_base, 215), (x_base + int(left_prog * 260), 235), (0, 200, 200), -1)
    cv2.putText(img, f"{left_elapsed:.1f}s", (x_base + 180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Right eye progress
    right_elapsed = 0.0
    if blink_state["right_start"] > 0.0:
        right_elapsed = now - blink_state["right_start"]
    right_prog = min(1.0, right_elapsed / max(0.0001, EYE_HOLD_THRESHOLD))
    cv2.putText(img, "Right Eye", (x_base, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(img, (x_base, 260), (w - 30, 280), (50, 50, 50), -1)
    cv2.rectangle(img, (x_base, 260), (x_base + int(right_prog * 260), 280), (0, 200, 200), -1)
    cv2.putText(img, f"{right_elapsed:.1f}s", (x_base + 180, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Both eyes progress
    both_elapsed = 0.0
    if blink_state["both_start"] > 0.0:
        both_elapsed = now - blink_state["both_start"]
    both_prog = min(1.0, both_elapsed / max(0.0001, EYE_HOLD_THRESHOLD))
    cv2.putText(img, "Both Eyes", (x_base, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(img, (x_base, 305), (w - 30, 325), (50, 50, 50), -1)
    cv2.rectangle(img, (x_base, 305), (x_base + int(both_prog * 260), 325), (0, 165, 255), -1)
    cv2.putText(img, f"{both_elapsed:.1f}s", (x_base + 180, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Device status
    cv2.putText(img, "DEVICE STATUS", (x_base, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Light1
    light1_color = (0, 255, 255) if state["light1"] else (50, 50, 50)
    cv2.circle(img, (x_base + 30, 390), 18, light1_color, -1)
    cv2.putText(img, f"Light 1: {'ON' if state['light1'] else 'OFF'}", (x_base + 70, 397), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # Light2
    light2_color = (0, 255, 255) if state["light2"] else (50, 50, 50)
    cv2.circle(img, (x_base + 30, 440), 18, light2_color, -1)
    cv2.putText(img, f"Light 2: {'ON' if state['light2'] else 'OFF'}", (x_base + 70, 447), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # Fan
    fan_color = (0, 0, 255) if state["fan"] else (50, 50, 50)
    cv2.circle(img, (x_base + 30, 490), 18, fan_color, -1)
    cv2.putText(img, f"Fan: {'ON' if state['fan'] else 'OFF'}", (x_base + 70, 497), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Logs
    cv2.rectangle(img, (x_base, 540), (w - 30, 640), (50, 50, 50), -1)
    cv2.putText(img, "LOG:", (x_base + 10, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(img, state["last_action"], (x_base + 10, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Instructions
    cv2.putText(img, "[M] Toggle Mode", (x_base, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1)
    cv2.putText(img, "[Q] Quit", (x_base, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1)

    return img

def check_auto_switch(hand_detected, face_detected, now):
    """Automatically switch mode based on detections, with hold time to prevent flickering."""
    if hand_detected:
        target = "GESTURE"
    elif face_detected:
        target = "BLINK"
    else:
        target = None

    if target != auto_state["target_mode"]:
        auto_state["target_mode"] = target
        auto_state["target_start"] = now
    elif target is not None:
        if now - auto_state["target_start"] >= AUTO_SWITCH_HOLD:
            if state["mode"] != target:
                state["mode"] = target
                state["last_action"] = f"Auto switched to {target}"
                # Reset timers on switch
                blink_state["left_closed"] = blink_state["right_closed"] = blink_state["both_closed"] = False
                blink_state["left_start"] = blink_state["right_start"] = blink_state["both_start"] = 0.0
                gesture_state["last_gesture_time"] = now
                auto_state["target_start"] = now  # Reset to prevent immediate re-switch
    else:
        auto_state["target_mode"] = None

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

print("Starting Demo...")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # mirror
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process
    hand_results = hands.process(imgRGB)
    face_results = face_mesh.process(imgRGB)

    hand_detected = bool(hand_results.multi_hand_landmarks)
    face_detected = bool(face_results.multi_face_landmarks)

    # Automatic mode switching
    check_auto_switch(hand_detected, face_detected, time.time())

    # Mode based logic
    if state["mode"] == "GESTURE":
        if hand_results.multi_hand_landmarks:
            for handLms in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(handLms)

                action = perform_gesture_action(fingers)
                if action:
                    print("Gesture:", action)

    elif state["mode"] == "BLINK":
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        if face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
            print("Face detected")  # Debug

        triggered = process_eye_holds(face_landmarks, h, w)
        if triggered == "LEFT":
            state["light1"] = not state["light1"]
            state["last_action"] = f"Left Eye Hold -> Light1 {'ON' if state['light1'] else 'OFF'}"
            print(state["last_action"])
        elif triggered == "RIGHT":
            state["light2"] = not state["light2"]
            state["last_action"] = f"Right Eye Hold -> Light2 {'ON' if state['light2'] else 'OFF'}"
            print(state["last_action"])
        elif triggered == "BOTH":
            state["fan"] = not state["fan"]
            state["last_action"] = f"Both Eyes Hold -> Fan {'ON' if state['fan'] else 'OFF'}"
            print(state["last_action"])

    # Draw dashboard
    img = draw_dashboard(img, state, blink_state)

    cv2.imshow(WINDOW_NAME, img)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        # Toggle mode
        state["mode"] = "BLINK" if state["mode"] == "GESTURE" else "GESTURE"
        state["last_action"] = "Mode switched to " + state["mode"]
        # Reset blink timers
        blink_state["left_closed"] = blink_state["right_closed"] = blink_state["both_closed"] = False
        blink_state["left_start"] = blink_state["right_start"] = blink_state["both_start"] = 0.0

cap.release()
cv2.destroyAllWindows()