from flask import Flask, Response, jsonify
import cv2
import threading
from ultralytics import YOLO
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# 初始化 YOLO 模型
model = YOLO('yolo11n.pt')
confidence_threshold = 0.5

# 视频捕获设备
cap = cv2.VideoCapture(0)
frame = None
lock = threading.Lock()

def capture_video():
    global frame
    while True:
        ret, current_frame = cap.read()
        if not ret:
            continue
        with lock:
            frame = current_frame

@app.route('/video_feed')
def video_feed():
    def generate():
        global frame
        while True:
            with lock:
                if frame is None:
                    continue
                # 编码为 JPEG 格式并返回
                _, encoded_image = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """API Endpoint to return detection details as JSON."""
    global frame
    with lock:
        if frame is None:
            return jsonify({'error': 'No frame captured yet.'})

        # 检测并返回检测框及类别信息
        results = model.predict(source=frame,
                                save=False, 
                                conf=confidence_threshold, 
                                show=False,
                                imgsz=160)

        detections = results[0].boxes.data.cpu().numpy()
        detection_list = []
        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection
            if confidence >= confidence_threshold and int(cls) == 0:
                detection_list.append({
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'confidence': float(confidence),
                    'class': int(cls)
                })

        return jsonify(detection_list)

if __name__ == '__main__':
    # 开启视频捕获线程
    video_thread = threading.Thread(target=capture_video, daemon=True)
    video_thread.start()
    
    # 启动 Flask 服务
    app.run(host='0.0.0.0', port=5000)

