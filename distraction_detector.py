import cv2 
from ultralytics import YOLO 
import mediapipe as mp 
import time 
last_alert_time = 0
cooldown_secs = 30

# Keep track of loggings
from logger import log_distraction

# Load YOLOv8 model (use 'yolov8n.pt' or 'yolov8s.pt' depending on your system)
model = YOLO("yolov8n.pt")  # Make sure this file is in your working directory

# Init Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh 
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils 

# Webcam init
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
prev_time = 0

from redirector import show_popup

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for performance
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect objects with YOLO
    results = model(frame_resized, verbose=False)[0]
    distraction_flag = False 
    distraction_type = None 

    # check if any common distractions are present
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r # x1, y1, x2, y2, conf is the confidence, cls is the class id
        class_id = int(cls)
        label = model.names[class_id]
        if label in ["cell phone", "iphone" "tv", "laptop"]:
            distraction_flag = True 
            distraction_type = label
            if distraction_flag and distraction_type:
                now = time.time() 
                if now - last_alert_time > cooldown_secs:
                    log_distraction(event_type=distraction_type, source="YOLOv8/Mediapipe") # param1=sample for better readibility
                    show_popup(goal="Finish Focus System")
                    last_alert_time = now
            
            cv2.putText(frame_resized, f"Distraction: {label}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
    # Mediapipe face detection
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame) 

    if face_results.multi_face_landmarks:
        distraction_flag = False    # User is present and attentive
        for face_landmarks in face_results.multi_face_landmarks:
            drawing_utils.draw_landmarks(
                frame_resized,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
    else:
        distraction_flag = True 
        distraction_type = "No Face Detected"
        if distraction_flag and distraction_type: # Make sure distraction_flag is True before logging, and distraction_type is not None
            log_distraction(event_type=distraction_type, source="YOLOv8/Mediapipe")

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0 
    prev_time = curr_time 
    cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display status
    if distraction_flag:
        cv2.putText(frame_resized, f"⚠️ DISTRACTION: {distraction_type}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
    # show output
    cv2.imshow("Distraction Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()