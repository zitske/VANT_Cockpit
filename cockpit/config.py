import math
import os
from pathlib import Path

import cv2


WIDTH, HEIGHT = 720, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
OSD_COLOR = (0, 255, 0)
YOLO_MODEL_SOURCE = os.getenv("YOLO_WEIGHTS_PATH", "yolo11n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))


def parse_imgsz(value):
    value = value.strip()
    if "," in value:
        height_str, width_str = value.split(",", 1)
        return int(height_str), int(width_str)
    return int(value)


def make_stride_multiple(value, stride=32):
    return int(math.ceil(value / stride) * stride)


def normalize_imgsz(imgsz, stride=32):
    if isinstance(imgsz, tuple):
        height, width = imgsz
        return make_stride_multiple(height, stride), make_stride_multiple(width, stride)
    return make_stride_multiple(imgsz, stride)


YOLO_IMGSZ = normalize_imgsz(parse_imgsz(os.getenv("YOLO_IMGSZ", "192,320")))
YOLO_RUNTIME = os.getenv("YOLO_RUNTIME", "auto").lower()
YOLO_AUTO_EXPORT = os.getenv("YOLO_AUTO_EXPORT", "0") == "1"
YOLO_EXPORT_FORMAT = os.getenv("YOLO_EXPORT_FORMAT", "openvino").lower()
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
CAPTURE_DIR = Path(os.getenv("YOLO_CAPTURE_DIR", "captures"))
DISPLAY_FPS = float(os.getenv("DISPLAY_FPS", "30"))
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "10"))
DETECTION_HZ = float(os.getenv("DETECTION_HZ", "2"))


def has_gui_display():
    if os.getenv("FORCE_HEADLESS", "0") == "1":
        return False
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
