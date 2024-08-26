# 能用
import os
import cv2
import torch
from pathlib import Path
# from ultralytics import YOLOv5
from models.common import DetectMultiBackend

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# model = YOLOv5('best.pt')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
model = DetectMultiBackend("best.pt", device="", dnn=False, data=ROOT, fp16=False)
model.to('cpu')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    with torch.no_grad():
        results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            label = model.names[int(cls)]

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # cv2.imshow('YOLOv5 Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
