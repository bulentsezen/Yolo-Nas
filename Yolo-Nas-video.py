import torch
from super_gradients.training import models
import cv2
import numpy as np
import math

cap = cv2.VideoCapture("cars.mp4")

# device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))

model = models.get("yolo_nas_s", pretrained_weights="coco")

count = 0

while True:
    ret, frame = cap.read()
    count += 1

    result = list(model.predict(frame, conf=0.35))[0]
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
    confidences = result.prediction.confidence
    labels = result.prediction.labels.tolist()

    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1,y1,x2,y2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)

    cv2.imshow("video", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
