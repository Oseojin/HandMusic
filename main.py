import cv2
import mediapipe as mp
import pygame
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

pygame.mixer.init()
sound_files = {
    'C': 'sounds/C.mp3', 'D': 'sounds/D.mp3', 'E': 'sounds/E.mp3',
    'F': 'sounds/F.mp3', 'G': 'sounds/G.mp3', 'A': 'sounds/A.mp3',
    'B': 'sounds/B.mp3',
    'Snare': 'sounds/Snare.mp3',
    'HiHat': 'sounds/HiHat.mp3',
    'Kick': 'sounds/Kick.mp3'
}

sounds = {k: pygame.mixer.Sound(v) for k, v in sound_files.items()}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

drum_zones = {'Snare': (0.4, 0.6), 'HiHat': (0.2, 0.4), 'Kick': (0.6, 0.8)}

keyboard_width = 0.3
zone_width = keyboard_width / 7
keyboard_zones = {
    'C': (0.35, 0.35 + zone_width), 'D': (0.35 + zone_width, 0.35 + 2 * zone_width),
    'E': (0.35 + 2 * zone_width, 0.35 + 3 * zone_width),
    'F': (0.35 + 3 * zone_width, 0.35 + 4 * zone_width),
    'G': (0.35 + 4 * zone_width, 0.35 + 5 * zone_width),
    'A': (0.35 + 5 * zone_width, 0.35 + 6 * zone_width),
    'B': (0.35 + 6 * zone_width, 0.35 + 7 * zone_width)
}

previous_finger_positions = {'Left': [0] * 5, 'Right': [0] * 5}
previous_fist_y = {'Left': 1.0, 'Right': 1.0}

def is_fist(landmarks):
    return all(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20])

def detect_gesture(landmarks, hand_type, frame_height):
    fingers = [
        landmarks[4].y * frame_height,
        landmarks[8].y * frame_height,
        landmarks[12].y * frame_height,
        landmarks[16].y * frame_height,
        landmarks[20].y * frame_height
    ]

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape
    result = hands.process(frame_rgb)

    mode = 'piano'
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_type = hand_info.classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_fist(hand_landmarks.landmark):
                mode = 'drum'
                fist_y = hand_landmarks.landmark[9].y

                if fist_y > previous_fist_y[hand_type] + 0.02:
                    for note, (start, end) in drum_zones.items():
                        if start <= hand_landmarks.landmark[9].x <= end:
                            sounds[note].play()

                previous_fist_y[hand_type] = fist_y

                if mode == 'drum':
                    for note, (start, end) in drum_zones.items():
                        start_x = int(start * frame_width)
                        end_x = int(end * frame_width)
                        cv2.rectangle(frame, (start_x, 0), (end_x, frame_height), (0, 255, 0), 2)
                        cv2.putText(frame, note, (start_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif not is_fist(hand_landmarks.landmark):
                mode = 'piano'
                if mode == 'piano':
                    for note, (start, end) in keyboard_zones.items():
                        start_x = int(start * frame_width)
                        end_x = int(end * frame_width)
                        cv2.rectangle(frame, (start_x, 0), (end_x, frame_height), (255, 0, 0), 2)
                        cv2.putText(frame, note, (start_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    if result.multi_hand_landmarks:
                        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                            hand_type = hand_info.classification[0].label
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            fingers = detect_gesture(hand_landmarks.landmark, hand_type, frame_height)

                            for i, y_pos in enumerate(fingers):
                                x_pos = hand_landmarks.landmark[i * 4 + 4].x

                                for note, (start, end) in keyboard_zones.items():
                                    if start <= x_pos <= end and y_pos > previous_finger_positions[hand_type][i] + 10:
                                        sounds[note].play()

                            previous_finger_positions[hand_type] = fingers

    cv2.imshow("Hand Gesture Sound Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
