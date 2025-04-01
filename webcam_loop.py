# env/webcam_loop.py

import cv2 

def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        cv2.imshow("Webcam view", frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'): # press q to escape
            break 

    cap.release()
    cv2.destroyAllWindows()