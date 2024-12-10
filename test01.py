from flask import Flask, Response
import cv2

app = Flask(__name__)

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 是默认的摄像头设备ID

def generate_frames():
    while True:
        # 从摄像头读取一帧
        success, frame = cap.read()
        if not success:
            break
        else:
            # 将图像转换为JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                # 将JPEG图像转换为字节流并通过Response返回
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

