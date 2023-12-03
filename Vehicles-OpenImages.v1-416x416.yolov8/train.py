from torch.fx.experimental.unification import variables
from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
import gc
del variables
gc.collect()

torch.cuda.memory_summary(device=None, abbreviated=False)

# Load a model
# Different types of models:
# YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
if __name__ == '__main__':
    model = YOLO("yolov8m.yaml")  # build a new model from scratch

    # Use the model
    results = model.train(data="data.yaml", epochs=150)  # train the model