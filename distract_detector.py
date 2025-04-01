# agents/distract_detector.py

import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
from core.logger import log_distraction
from redirector import send_notification

model = YOLO("yolov8n.pt")
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
last_alert_time = 0
cooldown_secs = 30

def detect():
    global last_alert_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (640, 480))
        results = model(resized, verbose=False)[0]

        distraction_flag = False
        distraction_type = None

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            label = model.names[int(cls)]
            if label in ["cell phone", "tv", "laptop"]:
                distraction_flag = True
                distraction_type = label

                cv2.putText(resized, f"Distraction: {label}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mediapipe face detection
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        faces = face_mesh.process(rgb)
        
        if not faces.multi_face_landmarks:
            distraction_flag = True
            distraction_type = "No Face Detected"

        if distraction_flag and distraction_type:
            now = time.time()
            if now - last_alert_time > cooldown_secs:
                log_distraction(distraction_type, "YOLOv8/Mediapipe")
                send_notification("Your Current Focus Goal")
                last_alert_time = now

        cv2.imshow("Distraction Detection", resized)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()