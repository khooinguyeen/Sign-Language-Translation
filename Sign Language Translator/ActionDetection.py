import cv2 
import numpy as np
import os 
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from numpy.lib.function_base import extract

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for data, numpy arrays
DATA_PATH = os.path.join('Data')

# Action to detect
actions = np.array(['xin chao', 'cam on', 'toi yeu ban'])

# 30 videos worth of data
no_sequences = 30

# length of sequence: 30 frame
sequence_length = 30

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # read feed
                ret, frame = cap.read()

                # detection
                img, results = mediapipe_detection(frame, holistic)
                print(results)

                # draw landmarks
                draw_styled_landmarks(img, results)

                # # apply wait logic
                # if frame_num == 0:
                #     cv2.putText(img, 'STARTING COLLECTION', (120, 200),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                #     cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                #     cv2.imshow('OpenCV Feed', img)
                #     cv2.waitKey(500)
                # else:
                #     cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # show
                cv2.imshow('OpenCV Feed', img)

                # # export keypoints
                # keypoints = extract_keypoints(results)
                # npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # np.save(npy_path, keypoints)
          
                # break when type 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()

