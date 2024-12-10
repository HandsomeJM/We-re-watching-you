# 电子监工

# git提交

### 初始化本地仓库

```bash
cd /home/sunrise/yolo_dj
git init
```

`git init`：将当前目录初始化为一个 Git 仓库。初始化后，Git 会在项目目录中创建一个隐藏文件夹 `.git`，用于存储版本控制相关的信息。

### 添加文件到暂存区

```bash
git add .
```

`git add .`：将当前目录下的所有文件（包括子目录）添加到 Git 的暂存区（staging area）。暂存区是用于保存即将提交的文件的地方。

### 提交文件到本地仓库

```bash
git commit -m "Initial commit"
```

- `git commit`：将暂存区的文件保存到本地仓库。
- `-m "Initial commit"`：添加一条提交信息，用于描述当前提交的内容（如 “初始提交”）。

### 关联远程仓库

```bash
git remote add origin git@github.com:HandsomeJM/We-re-watching-you.git
```

- `git remote add`：添加一个远程仓库。
- `origin`：为这个远程仓库起的名字（一般用 `origin`，可以自定义）。
- `git@github.com:HandsomeJM/We-re-watching-you.git`：这是远程仓库的地址，通过 SSH 方式连接。

### 推送代码到远程仓库

```bash
git branch -M main
git push -u origin main
```

- `git branch -M main`：
    - 将当前分支重命名为 `main`。这是 Git 的默认主分支名称（老版本是 `master`）。
- `git push -u origin main`：
    - 将本地的 `main` 分支推送到远程仓库的 `main` 分支。
    - `u`：设置远程分支为默认上游分支，以后只需运行 `git push`。

# 代码

## 在`fuwu3.py`这个文件

内容如下

```bash
from flask import Flask, Response, jsonify, request
import cv2
import torch
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO('yolo11n.pt')
confidence_threshold = 0.5
detections_list = []

def generate_frames():
    global detections_list
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
        detections_list = []

        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection
            if confidence >= confidence_threshold and int(cls) == 0:
                confidence = float(confidence)
                detections_list.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": confidence
                })
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections', methods=['GET'])
def get_detections():
    return jsonify(detections_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

# conda打包

## **使用 `conda pack` 打包**

```bash
conda install conda-pack
```

安装完成后，使用以下命令打包

```bash
conda pack -n your_env_name -o your_env_name.tar.gz
```

虚拟环境已经打包放在了目录中