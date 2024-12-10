import cv2
import torch
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(0)  
confidence_threshold = 0.5 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame,
                            model="yolo11n.pt",
                            save=False, 
                            conf=confidence_threshold, 
                            show=False,
                            imgsz=160)

    detections = results[0].boxes.data.cpu().numpy() 
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if confidence >= confidence_threshold and int(cls) == 0:
            confidence = f"{confidence:.2f}"
            print(x1, y1, x2, y2, confidence)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-time Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
