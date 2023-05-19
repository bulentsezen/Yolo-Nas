import torch
from super_gradients.training import models

model_nas = models.get("yolo_nas_s", pretrained_weights="coco")

model = model_nas.to("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

model.predict_webcam()
