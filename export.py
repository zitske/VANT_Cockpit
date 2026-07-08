from ultralytics import YOLO

from cockpit.config import YOLO_IMGSZ, YOLO_MODEL_SOURCE


model = YOLO(YOLO_MODEL_SOURCE, task="detect")
model.export(format="onnx", imgsz=YOLO_IMGSZ, half=True)