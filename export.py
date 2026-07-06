from ultralytics import YOLO

# Mesma resolução base usada no main.py: PAL SD 720x576
WIDTH, HEIGHT = 720, 576

# Carrega o modelo (ex: 'yolov11n.pt')
model = YOLO('yolov11n.pt')

# Exporta usando a resolução do app principal.
# Ultralytics espera (altura, largura) ao receber tuple.
model.export(format='onnx', imgsz=(HEIGHT, WIDTH), half=True)