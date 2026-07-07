from ultralytics import YOLO

# Mesma resolução base usada no main.py: 16:9
WIDTH, HEIGHT = 1280, 720


def make_stride_multiple(value, stride=32):
	return int(((value + stride - 1) // stride) * stride)


YOLO_WIDTH = make_stride_multiple(WIDTH)

# Carrega o modelo (ex: 'yolo11n.pt')
model = YOLO('yolo11n.pt')

# Exporta usando a resolução do app principal.
# Ultralytics espera (altura, largura) ao receber tuple.
model.export(format='onnx', imgsz=(HEIGHT, YOLO_WIDTH), half=True)