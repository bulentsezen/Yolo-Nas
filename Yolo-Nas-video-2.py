import torch
from super_gradients.training import models
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

model = models.get("yolo_nas_s", pretrained_weights="coco")

count = 0

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))

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

        classname = int(cls)
        class_name = classNames[classname]

        conf = math.ceil((confidence*100))/100
        label = f'{class_name}{conf}'
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = x1 + t_size[0], y1-t_size[1]-3

        cv2.rectangle(frame, (x1, y1), c2, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1,y1-2), 0,1, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        out.write(frame)

    cv2.imshow("video", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

out.release
cap.release
cv2.destroyAllWindows()