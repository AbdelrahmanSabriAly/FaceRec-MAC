# Import necessary libraries
import os
import time
import cv2
import pandas as pd
import numpy as np
import pickle
from datetime import date
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gdown
from tqdm import tqdm



# Threshold for cosine similarity used for face recognition
COSINE_THRESHOLD = 0.5


# Function to load the face detection and recognition models, and the user dictionary
def load_models():
    # Initialize models for face detection & recognition
    weights = os.path.join( "./models",
                            "face_detection_yunet_2023mar_int8.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    face_detector.setScoreThreshold(0.87)

    weights = os.path.join( "./models", "face_recognition_sface_2021dec_int8.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    return face_detector, face_recognizer

face_detector, face_recognizer = load_models()


# Function to match the given feature with the features in the dictionary using the recognizer
def match( feature1,dictionary):
    max_score = 0.0
    sim_user_id = ""
    # Loop through each user in the dictionary and find the one with the highest similarity score
    for user_id, feature2 in zip(dictionary.keys(), dictionary.values()):
        score = face_recognizer.match(
            feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            sim_user_id = user_id
    # Check if the maximum similarity score is above the threshold for authentication
    if max_score < COSINE_THRESHOLD:
        return False, ("", 0.0)
    return True, (sim_user_id, max_score)

# Function to recognize faces in the given image using the face detector and recognizer
def recognize_face(image,file_name = None):
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0),
                           fx=500 / image.shape[0], fy=500 / image.shape[0])

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))
    try:
        # Detect faces in the image using the face detector
        dts = time.time()
        _, faces = face_detector.detect(image)
        # Check if the image contains a face (if file_name is provided, otherwise it's not required)
        if file_name is not None:
            assert len(faces) > 0, f'the file {file_name} has no face'

        faces = faces if faces is not None else []
        features = []
        print(f'time detection  = {time.time() - dts}')
        # Extract features for each detected face
        for face in faces:
            rts = time.time()

            aligned_face = face_recognizer.alignCrop(image, face)
            feat = face_recognizer.feature(aligned_face)
            print(f'time recognition  = {time.time() - rts}')

            features.append(feat)
        return features, faces
    except Exception as e:
        print(e)
        print(file_name)
        return None, None


def face_recognition_process(image,dictionary):
    fetures, faces = recognize_face(image, face_detector, face_recognizer)
    for idx, (face, feature) in enumerate(zip(faces, fetures)):
        result, user = match(face_recognizer, feature, dictionary)
        box = list(map(int, face[:4]))
        color = (0, 255, 0) if result else (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

        id_name, score = user if result else (f"unknown_{idx}", 0.0)
        text = "{0} ({1:.2f})".format(id_name, score)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        cv2.putText(image, text, position, font, scale,
                    color, thickness, cv2.LINE_AA)
    return image



def get_attendance_list(id_list):
    df = pd.read_excel("all.xlsx")
    today = date.today()
    df[today] = 0

    for id in id_list:
        df.loc[df['ID'] == id,today] = 1
    return df

