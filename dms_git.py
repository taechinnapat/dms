import os
import cv2
import numpy as np
import time
import mediapipe as mp
from scipy.spatial import distance
import threading
import pygame
import requests
import base64
import firebase_admin
from firebase_admin import credentials, firestore

# ตั้งค่าพารามิเตอร์
ALERT_FOLDER = 'alert_images'
EAR_THRESHOLD = 0.16
MAR_THRESHOLD = 0.6
CLOSED_EYE_FRAMES = 60
PHONE_NEAR_EAR_THRESHOLD = 0.1
CIGARETTE_NEAR_MOUTH_THRESHOLD = 0.1
BOTTLE_NEAR_MOUTH_THRESHOLD = 0.1
MAX_YAWN_COUNT = 5
YAWN_ALERT_INTERVAL = 60

# Firebase setup
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

USER_EMAIL = "techinnapat@gmail.com"

# GitHub setup
GITHUB_API_URL = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"
GITHUB_TOKEN = "your_personal_access_token"  # Replace with your GitHub token
OWNER = "your_github_username"  # Replace with your GitHub username
REPO = "your_repository_name"  # Replace with your GitHub repository name
FOLDER_PATH = "alert_images"  # Folder path in the repository to store the images

# Google Drive Connect (Not used anymore)
# from googleapiclient.discovery import build
# from google.oauth2 import service_account
# from googleapiclient.http import MediaFileUpload

# Function to upload photo to GitHub
def upload_photo_to_github(file_path):
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as file:
        encoded_file_content = base64.b64encode(file.read()).decode('utf-8')

    # Prepare the payload for GitHub API
    payload = {
        "message": f"Upload {file_name}",
        "content": encoded_file_content
    }

    # Make the request to GitHub API
    url = GITHUB_API_URL.format(owner=OWNER, repo=REPO, path=f"{FOLDER_PATH}/{file_name}")
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}"
    }
    
    response = requests.put(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"Uploaded {file_path} to GitHub")
        return response.json()['content']['download_url']  # Return file URL from GitHub
    else:
        print(f"Error uploading {file_path} to GitHub: {response.status_code}, {response.text}")
        return None

# Function to upload alert data to Firebase
def upload_to_firebase(file_path, alert_type):
    github_url = upload_photo_to_github(file_path)  # Upload to GitHub
    if github_url:
        alert_data = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "alert_type": alert_type,
            "image_url": github_url,  # Use URL from GitHub
            "user_email": USER_EMAIL
        }
        db.collection("alerts").add(alert_data)  # Upload to Firebase
        print(f"Uploaded alert to Firebase: {alert_type} with image URL {github_url}")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
pygame.mixer.init()

# Function to play alert sound
def play_alert_sound(file_name):
    sound = pygame.mixer.Sound(file_name)
    sound.play()

# Function to capture frame and upload
def capture_frame(frame, alert_type):
    if not os.path.exists(ALERT_FOLDER):
        os.makedirs(ALERT_FOLDER)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{alert_type}_{timestamp}.jpg"
    file_path = os.path.join(ALERT_FOLDER, filename)
    cv2.imwrite(file_path, frame)
    
    # Upload file to GitHub and Firebase asynchronously
    threading.Thread(target=upload_to_firebase, args=(file_path, alert_type)).start()
    print(f"Captured frame for {alert_type} alert as {file_path}")

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate MAR (Mouth Aspect Ratio)
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[3], mouth_points[5])
    B = distance.euclidean(mouth_points[2], mouth_points[6])
    C = distance.euclidean(mouth_points[1], mouth_points[7])
    D = distance.euclidean(mouth_points[0], mouth_points[4])
    return (A + B + C) / (2.0 * D)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

ptime = 0
eye_closed_start = None
drowsiness_alert_played = False
yawning_alert_played = False
phone_alert_played = False
cigarette_alert_played = False
bottle_alert_played = False
yawn_count = 0
yawn_start_time = None
last_alert_time = None

# Main loop for capturing video frames and detecting alerts
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    phone_position = None
    cigarette_position = None
    bottle_position = None

    # MediaPipe face detection
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]
            left_ear_landmark = 234  
            right_ear_landmark = 454

            # Extract eye and mouth points
            left_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in left_eye_indices]
            right_eye = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in right_eye_indices]
            mouth_points = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in mouth_indices]
            left_ear_coords = (int(landmarks[left_ear_landmark].x * frame.shape[1]), int(landmarks[left_ear_landmark].y * frame.shape[0]))
            right_ear_coords = (int(landmarks[right_ear_landmark].x * frame.shape[1]), int(landmarks[right_ear_landmark].y * frame.shape[0]))

            # Calculate EAR and MAR
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = calculate_mar(mouth_points)

            # Display EAR and MAR on the screen
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Drowsiness detection
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif (time.time() - eye_closed_start) > CLOSED_EYE_FRAMES / video_capture.get(cv2.CAP_PROP_FPS):
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not drowsiness_alert_played:
                        threading.Thread(target=play_alert_sound, args=('AlertSound/drowsiness_alert.wav',)).start()
                        capture_frame(frame, "มีอาการง่วงในขณะขับรถ")
                        drowsiness_alert_played = True
            else:
                eye_closed_start = None
                drowsiness_alert_played = False

            if mar > MAR_THRESHOLD:
                if not yawning_alert_played:
                    yawning_alert_played = True
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    elapsed_time = time.time() - yawn_start_time
                    if elapsed_time <= YAWN_ALERT_INTERVAL:
                        yawn_count += 1
                        if yawn_count >= MAX_YAWN_COUNT:
                            if last_alert_time is None or time.time() - last_alert_time > 0.5:
                                threading.Thread(target=play_alert_sound, args=('AlertSound/yawning_alert.wav',)).start()
                                capture_frame(frame, "มีอาการหาวในขณะขับรถ")
                                last_alert_time = time.time()
                    else:
                        yawn_count = 0
                        yawn_start_time = None
            else:
                yawning_alert_played = False

            if phone_position:
                distance_left = distance.euclidean(phone_position, left_ear_coords)
                distance_right = distance.euclidean(phone_position, right_ear_coords)
                # print(f"{distance_right} < {PHONE_NEAR_EAR_THRESHOLD} ")
                if distance_left < PHONE_NEAR_EAR_THRESHOLD * frame.shape[1] or distance_right < PHONE_NEAR_EAR_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "PHONE ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not phone_alert_played:
                        print(threading.Thread(target=play_alert_sound, args=('phone_detected.wav',)).start())
                        capture_frame(frame, "ห้ามใช้โทรศัพท์ในขณะขับรถ")
                        phone_alert_played = True
                else:
                    phone_alert_played = False


            if cigarette_position:
                distance_mouth = distance.euclidean(cigarette_position,mouth_points[7])
                if distance_mouth < CIGARETTE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "CIGARETTE ALERT!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not cigarette_alert_played:
                        threading.Thread(target=play_alert_sound, args=('cigarette_detected.wav',)).start()
                        capture_frame(frame, "สูบบุหรี่ในขณะขับรถ")
                        cigarette_alert_played = True
                else:
                    cigarette_alert_played = False

            if bottle_position:
                distance_mouth1 = distance.euclidean(bottle_position,mouth_points[7])
                # print(distance_mouth1)
                if distance_mouth1 < BOTTLE_NEAR_MOUTH_THRESHOLD * frame.shape[1]:
                    cv2.putText(frame, "DRINKING WATER ALERT!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if not bottle_alert_played:
                        threading.Thread(target=play_alert_sound, args=('bottle_detected.wav',)).start()
                        capture_frame(frame, "ดื่มน้ำในขณะขับรถ")
                        bottle_alert_played = True
                else:
                    bottle_alert_played = False

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Drowsiness Detection with YOLO', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()