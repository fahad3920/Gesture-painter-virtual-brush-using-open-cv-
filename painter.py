import cv2
import numpy as np
import mediapipe as mp

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Colors
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
color_index = 0
draw_color = colors[color_index]

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Drawing canvas
canvas = np.zeros((720, 1280, 3), np.uint8)

def fingers_up(hand_landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if hand_landmarks.landmark[tips[id]].y < hand_landmarks.landmark[tips[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lm_list = handLms.landmark

        index_x = int(lm_list[8].x * 1280)
        index_y = int(lm_list[8].y * 720)

        fingers = fingers_up(handLms)

        # Selection Mode – Two Fingers
        if fingers[1] and fingers[2]:
            prev_x, prev_y = 0, 0
            if index_y < 100:
                if 200 < index_x < 300:
                    color_index = 0
                elif 350 < index_x < 450:
                    color_index = 1
                elif 500 < index_x < 600:
                    color_index = 2
                elif 650 < index_x < 750:
                    color_index = 3
                draw_color = colors[color_index]
            cv2.rectangle(img, (index_x - 25, index_y - 25), (index_x + 25, index_y + 25), draw_color, cv2.FILLED)

        # Drawing Mode – Index Finger Up
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (index_x, index_y), 15, draw_color, cv2.FILLED)
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = index_x, index_y
            cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), draw_color, 10)
            prev_x, prev_y = index_x, index_y

        # Clear Canvas – Pinky finger up only
        elif fingers == [0, 0, 0, 0, 1]:
            canvas = np.zeros((720, 1280, 3), np.uint8)

    # Merge canvas and webcam image
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Draw header
    cv2.rectangle(img, (200, 0), (750, 100), (0, 0, 0), -1)
    cv2.putText(img, 'Select Color', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for i in range(len(colors)):
        cv2.rectangle(img, (200 + i * 150, 10), (300 + i * 150, 90), colors[i], -1)

    cv2.imshow("Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
