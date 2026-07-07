from ultralytics import YOLO

# Mesma resolução base usada no main.py: resolução nativa
WIDTH, HEIGHT = 720, 480


def make_stride_multiple(value, stride=32):
	return int(((value + stride - 1) // stride) * stride)


YOLO_WIDTH = make_stride_multiple(WIDTH)

# Resolução de inferência leve para Raspberry Pi 5.
YOLO_INFER_HEIGHT, YOLO_INFER_WIDTH = 192, 320

# Carrega o modelo de deteccao explicitamente.
model = YOLO('yolo11n.pt', task='detect')

# Exporta usando a resolução do app principal.
# Ultralytics espera (altura, largura) ao receber tuple.
model.export(format='onnx', imgsz=(YOLO_INFER_HEIGHT, YOLO_INFER_WIDTH), half=True)