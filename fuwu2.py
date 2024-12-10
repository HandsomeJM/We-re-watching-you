from flask import Flask, Response
import cv2
import torch
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO('yolo11n.pt')
confidence_threshold = 0.4


def generate_frames():
    cap = cv2.VideoCapture(0) 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        

        results = model.predict(source=frame,
                                save=False, 
                                conf=confidence_threshold, 
                                show=False,
                                imgsz=160)
        detections = results[0].boxes.data.cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection
            if confidence >= confidence_threshold and int(cls) == 0:
                confidence = f"{confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
