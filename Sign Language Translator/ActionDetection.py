import cv2 
import numpy as np
import os 
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # draw utilities

def mediapipe_detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img.flags.writable = False
    results = model.process(img)
    # img.flags.writable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results

def draw_landmarks(img, results):
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(img, results):
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color = (80, 110, 10), thickness = 0, circle_radius = 0),
                            mp_drawing.DrawingSpec(color = (80, 256, 121), thickness = 1, circle_radius = 1)
                            )
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color = (80, 22, 10), thickness = 2, circle_radius = 4),
                            mp_drawing.DrawingSpec(color = (80, 44, 121), thickness = 2, circle_radius = 2)
                            )
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4),
                            mp_drawing.DrawingSpec(color = (121, 44, 250), thickness = 2, circle_radius = 2)
                            )
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2, circle_radius = 4),
                            mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius = 2)
                            )
# mp_holistic.POSE_CONNECTIONS

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        # read feed
        ret, frame = cap.read()

        # detection
        img, results = mediapipe_detection(frame, holistic)
        print(results)

        # draw landmarks
        draw_styled_landmarks(img, results)

        # show
        cv2.imshow('OpenCV Feed', img)

        # break when tap 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

