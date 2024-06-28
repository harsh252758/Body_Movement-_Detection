from django.shortcuts import render
from django.http import JsonResponse
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def index(request):
    return render(request, 'index.html')

def process_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, f"Angle: {angle:.2f}", 
                            tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                _, buffer = cv2.imencode('.jpg', image)
                image_encoded = base64.b64encode(buffer).decode('utf-8')

                return JsonResponse({'image': image_encoded, 'angle': angle})
            else:
                return JsonResponse({'error': 'No pose landmarks detected'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)
