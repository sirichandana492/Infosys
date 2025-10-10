import cv2
import pyautogui
import mediapipe as mp
import math
import matplotlib.pyplot as plt
from collections import deque

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Volume and distance storage for graph
volume_history = deque(maxlen=50)
distance_history = deque(maxlen=50)

# Matplotlib setup
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], 'b-', label="Volume (%)")
ax.set_ylim(0, 100)
ax.set_xlim(0, 50)
ax.set_xlabel("Frames")
ax.set_ylabel("Volume (%)")
ax.legend()
plt.title("Volume Level Based on Distance")

prev_level = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    dist = 0
    volume_level = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb tip (id=4) and index tip (id=8)
            x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

            # Draw markers
            cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculate distance
            dist = math.hypot(x2 - x1, y2 - y1)

            # Map distance to volume level (0–100)
            volume_level = int(((dist - 20) / (200 - 20)) * 100)
            volume_level = max(0, min(volume_level, 100))  # Limit 0–100

            # Show distance text
            cv2.putText(frame, f"Dist: {int(dist)} px", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw volume bar
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            bar_height = int(400 - (volume_level * 2.5))
            cv2.rectangle(frame, (50, bar_height), (85, 400), (0, 255, 0), -1)
            cv2.putText(frame, f'Vol: {volume_level}%', (40, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Control volume using pyautogui
            if prev_level is None or abs(volume_level - prev_level) > 5:
                if volume_level > (prev_level if prev_level else 0):
                    pyautogui.press("volumeup")
                else:
                    pyautogui.press("volumedown")
                prev_level = volume_level

    # Update graph data
    volume_history.append(volume_level)
    distance_history.append(dist)
    line1.set_data(range(len(volume_history)), volume_history)
    ax.set_xlim(0, max(50, len(volume_history)))
    fig.canvas.draw()
    fig.canvas.flush_events()

    cv2.imshow("Volume Mapping & Control + Graph", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        print("\nExiting program...")
        break

cap.release()
cv2.destroyAllWindows()
plt.close(fig)
