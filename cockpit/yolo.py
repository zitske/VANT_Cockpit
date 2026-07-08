from datetime import datetime
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .config import (
    CAPTURE_DIR,
    OSD_COLOR,
    YOLO_AUTO_EXPORT,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_EXPORT_FORMAT,
    YOLO_IMGSZ,
    YOLO_MODEL_SOURCE,
    YOLO_RUNTIME,
)

YOLO_ACTIVE_BACKEND = "pt"


def load_person_detector():
    global YOLO_ACTIVE_BACKEND

    if YOLO is None:
        print("Aviso: ultralytics nao esta instalado; deteccao YOLO desativada.")
        return None

    try:
        source = Path(YOLO_MODEL_SOURCE)

        if source.exists() and source.suffix.lower() == ".pt":
            onnx_source = source.with_suffix(".onnx")
            openvino_source = source.with_name(f"{source.stem}_openvino_model")

            if YOLO_RUNTIME == "openvino":
                runtime_order = ["openvino", "onnx", "pt"]
            elif YOLO_RUNTIME == "onnx":
                runtime_order = ["onnx", "openvino", "pt"]
            else:
                runtime_order = ["openvino", "onnx", "pt"]

            backend_sources = {
                "openvino": openvino_source,
                "onnx": onnx_source,
                "pt": source,
            }

            selected_backend = "pt"
            selected_source = source

            for backend in runtime_order:
                candidate = backend_sources[backend]
                if backend == "pt" or candidate.exists():
                    selected_backend = backend
                    selected_source = candidate
                    break

            if selected_backend == "pt" and YOLO_AUTO_EXPORT and YOLO_EXPORT_FORMAT in ("onnx", "openvino"):
                try:
                    print(f"YOLO: exportando {source.name} para {YOLO_EXPORT_FORMAT}...")
                    export_model = YOLO(str(source), task="detect")
                    export_model.export(format=YOLO_EXPORT_FORMAT, imgsz=YOLO_IMGSZ)
                    exported_source = backend_sources[YOLO_EXPORT_FORMAT]
                    if exported_source.exists():
                        selected_backend = YOLO_EXPORT_FORMAT
                        selected_source = exported_source
                except Exception as exc:
                    print(f"Aviso: falha ao exportar YOLO para {YOLO_EXPORT_FORMAT} ({exc}); seguindo com PT.")

            YOLO_ACTIVE_BACKEND = selected_backend
            print(f"YOLO backend ativo: {YOLO_ACTIVE_BACKEND} ({selected_source})")
            return YOLO(str(selected_source), task="detect")

        if source.exists():
            if source.suffix.lower() == ".onnx":
                YOLO_ACTIVE_BACKEND = "onnx"
            elif source.is_dir() and source.name.endswith("_openvino_model"):
                YOLO_ACTIVE_BACKEND = "openvino"
            else:
                YOLO_ACTIVE_BACKEND = "pt"

            print(f"YOLO backend ativo: {YOLO_ACTIVE_BACKEND} ({source})")
            return YOLO(str(source), task="detect")

        YOLO_ACTIVE_BACKEND = "pt"
        print(f"YOLO backend ativo: {YOLO_ACTIVE_BACKEND} ({YOLO_MODEL_SOURCE})")
        return YOLO(YOLO_MODEL_SOURCE, task="detect")
    except Exception as exc:
        print(f"Aviso: nao foi possivel carregar YOLO em {YOLO_MODEL_SOURCE} ({exc}); deteccao desativada.")
        return None


def detect_persons(frame, detector):
    if detector is None or frame is None:
        return []

    try:
        predict_kwargs = {
            "conf": YOLO_CONFIDENCE,
            "imgsz": YOLO_IMGSZ,
            "classes": [0],
            "verbose": False,
        }

        if YOLO_ACTIVE_BACKEND == "pt":
            predict_kwargs["device"] = YOLO_DEVICE

        try:
            results = detector.predict(frame, **predict_kwargs)
        except TypeError:
            predict_kwargs.pop("device", None)
            results = detector.predict(frame, **predict_kwargs)

        if not results:
            return []

        detections = []
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        for index, box in enumerate(boxes.xyxy.cpu().numpy(), start=1):
            x1, y1, x2, y2 = [int(value) for value in box]
            person_x = int((x1 + x2) / 2)
            person_y = int((y1 + y2) / 2)
            offset_x = person_x - center_x
            offset_y = person_y - center_y
            detections.append(
                {
                    "index": index,
                    "bbox": (x1, y1, x2, y2),
                    "center": (person_x, person_y),
                    "offset": (offset_x, offset_y),
                }
            )

        return detections
    except Exception as exc:
        print(f"Aviso: falha na deteccao YOLO ({exc}); seguindo sem overlay.")
        return []


def draw_person_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        person_x, person_y = det["center"]
        offset_x, offset_y = det["offset"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), OSD_COLOR, 2)
        label = f"P{det['index']} X:{person_x} Y:{person_y} dX:{offset_x:+d} dY:{offset_y:+d}"
        label_y = max(20, y1 - 10)
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, OSD_COLOR, 2)


def save_person_snapshot(scene, detections, capture_dir=CAPTURE_DIR):
    if not detections:
        return None

    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = capture_dir / f"person_{timestamp}.jpg"

    grayscale_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(filename), grayscale_scene, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return filename
