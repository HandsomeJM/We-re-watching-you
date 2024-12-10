import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("yolo11n.onnx")

cap = cv2.VideoCapture(0)
confidence_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))     
    img_input = np.transpose(img_resized, (2, 0, 1)).astype(np.float32) 
    img_input = np.expand_dims(img_input, axis=0)  

    inputs = {session.get_inputs()[0].name: img_input}
    outputs = session.run(None, inputs)
       
    boxes = outputs[0] 
    print(boxes.shape)
    boxes = boxes[0]
    print(boxes.shape)
    for box in boxes:
        x1, y1, x2, y2, confidence, *cls = box
        if confidence >= confidence_threshold:
            label = f"Class {cls}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-time Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

