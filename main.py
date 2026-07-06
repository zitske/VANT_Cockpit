import cv2
import numpy as np
import math
import time
import os
import glob
import threading
from queue import Queue, Empty, Full
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import tkinter as tk
except ImportError:
    tk = None

# --- Configurações da Interface ---
WIDTH, HEIGHT = 720, 576 # Resolução PAL SD
FONT = cv2.FONT_HERSHEY_SIMPLEX
OSD_COLOR = (0, 255, 0) # Verde clássico de FPV
YOLO_MODEL_SOURCE = os.getenv("YOLO_WEIGHTS_PATH", "yolov11n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "320"))
YOLO_RUNTIME = os.getenv("YOLO_RUNTIME", "auto").lower()
YOLO_AUTO_EXPORT = os.getenv("YOLO_AUTO_EXPORT", "0") == "1"
YOLO_EXPORT_FORMAT = os.getenv("YOLO_EXPORT_FORMAT", "openvino").lower()
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
CAPTURE_DIR = Path(os.getenv("YOLO_CAPTURE_DIR", "captures"))
DISPLAY_FPS = float(os.getenv("DISPLAY_FPS", "30"))
TELEMETRY_HZ = float(os.getenv("TELEMETRY_HZ", "10"))
DETECTION_HZ = float(os.getenv("DETECTION_HZ", "2"))
YOLO_ACTIVE_BACKEND = "pt"


def resolve_camera_source(preferred_index, by_id_pattern):
    matches = sorted(glob.glob(by_id_pattern))
    if matches:
        return matches[0]
    return preferred_index


def open_camera(preferred_index, by_id_pattern):
    candidates = []

    for match in sorted(glob.glob(by_id_pattern)):
        candidates.append(match)
        real_path = os.path.realpath(match)
        if real_path != match:
            candidates.append(real_path)

    candidates.append(preferred_index)

    for source in candidates:
        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture, source
            capture.release()

    return None, candidates[0] if candidates else preferred_index


def put_latest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)


def get_latest_or_last(q, last_value):
    latest = last_value
    while True:
        try:
            latest = q.get_nowait()
        except Empty:
            break
    return latest


def create_error_frame(shape, error_text, color=(50, 50, 50)):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :] = color
    cv2.putText(frame, error_text, (20, shape[0] // 2), FONT, 0.8, (0, 0, 255), 2)
    return frame


def capture_worker(capture, output_queues, stop_event):
    while not stop_event.is_set():
        ret, frame = capture.read()
        if not ret:
            time.sleep(0.005)
            continue

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for q in output_queues:
            put_latest(q, frame)


def yolo_inference_worker(normal_queue, thermal_queue, detector, shared_state, state_lock, stop_event):
    last_run = 0.0
    latest_normal = None
    latest_thermal = None

    while not stop_event.is_set():
        latest_normal = get_latest_or_last(normal_queue, latest_normal)
        latest_thermal = get_latest_or_last(thermal_queue, latest_thermal)

        with state_lock:
            enabled = shared_state["person_detection_enabled"]

        if not enabled or detector is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_run < 1.0 / DETECTION_HZ:
            time.sleep(0.005)
            continue

        normal_detections = detect_persons(latest_normal, detector) if latest_normal is not None else []
        thermal_detections = detect_persons(latest_thermal, detector) if latest_thermal is not None else []

        with state_lock:
            shared_state["normal_detections"] = normal_detections
            shared_state["thermal_detections"] = thermal_detections

        last_run = now

# --- Variáveis de Estado da Simulação ---
# Dicionário para guardar todos os nossos dados simulados
sim_data = {
    "pitch": 0.0,      # Graus
    "roll": 0.0,       # Graus
    "heading": 0.0,    # Graus (0-360)
    "airspeed": 0.0,   # m/s
    "altitude": 0.0,   # metros
    "ground_speed": 0.0, # m/s
    "lat": -22.9068,   # Coordenadas (ex: Rio)
    "lon": -43.1729,
    "home_lat": -22.9068,
    "home_lon": -43.1729,
    "sats": 10,        # Número de satélites GPS
    "batt_volt": 16.8, # Voltagem (ex: 4S)
    "flight_mode": "STABILIZE",
    "thermal_is_main": True # Estado de troca de câmera
}


def get_display_resolution(default_width=WIDTH, default_height=HEIGHT):
    if tk is None:
        return default_width, default_height

    try:
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        if screen_width > 0 and screen_height > 0:
            return screen_width, screen_height
    except Exception:
        pass

    return default_width, default_height


def read_capture_frame(capture, fallback_shape, error_text, loop=False):
    if capture:
        ret, frame = capture.read()

        if not ret and loop:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = capture.read()

        if ret:
            return frame

    frame = np.zeros(fallback_shape, dtype=np.uint8)
    frame[:, :] = (50, 50, 50)
    cv2.putText(frame, error_text, (20, fallback_shape[0] // 2), FONT, 0.8, (0, 0, 255), 2)
    return frame

# --- Funções de Desenho da Interface ---


# A função agora recebe cap_normal e cap_thermal
def create_simulated_frames(t, cap_normal, cap_thermal):
    """Lê um frame de ambos os vídeos de simulação."""

    frame_normal = read_capture_frame(
        cap_normal,
        (480, 640, 3),
        "CAMERA 0 ERROR",
        loop=False,
    )

    frame_thermal = read_capture_frame(
        cap_thermal,
        (192, 256, 3),
        "CAMERA 2 ERROR",
        loop=False,
    )

    if len(frame_thermal.shape) == 2 or frame_thermal.shape[2] == 1:
        frame_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_GRAY2BGR)

    return frame_normal, frame_thermal
def draw_artificial_horizon(canvas, roll_deg, pitch_deg, cx, cy, radius):
    """
    Desenha um Horizonte Artificial (AHI) completo como um overlay transparente,
    cobindo a tela toda com as linhas de pitch e indicador de roll.
    """
    
    # Cores
    # SKY_COLOR = (255, 128, 0) # Não mais usado
    # GROUND_COLOR = (0, 80, 130) # Não mais usado
    LINE_COLOR = OSD_COLOR # Usar a cor OSD para as linhas

    # Converter para radianos
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    
    # --- Parâmetros de Redesenho para tela cheia ---
    # Agora o AHI será desenhado diretamente sobre o 'canvas' principal
    # Usaremos WIDTH e HEIGHT da tela inteira como referência
    
    # O centro de rotação e escala será o centro da tela
    full_cx, full_cy = canvas.shape[1] // 2, canvas.shape[0] // 2
    
    # A "escala" do pitch será baseada na altura total da tela
    # Ajuste 'pixels_per_degree' para controlar o "zoom" do pitch
    pixels_per_degree = 4 # Valor ajustado para preencher a tela verticalmente
    
    # Calcular o deslocamento vertical do pitch
    pitch_shift_y = int(pitch_deg * pixels_per_degree)

    # Criar um canvas temporário para as linhas de pitch, para que possamos rotacioná-lo
    # Este canvas será do tamanho da tela principal para o overlay
    overlay = np.zeros_like(canvas) # Cria uma cópia preta com as mesmas dimensões

    # --- Desenhar a Linha Central do Horizonte (0 graus pitch) ---
    horizon_center_y = full_cy - pitch_shift_y
    
    # Desenhar as linhas de pitch (branco)
    # Linha central (0 graus)
    cv2.line(overlay, (0, horizon_center_y), (canvas.shape[1], horizon_center_y), LINE_COLOR, 2)
    
    # --- Desenhar a Escala de Pitch ---
    # Linhas de +10, +20, +30, etc. (acima do horizonte)
    for p in range(10, 91, 10): # Linhas positivas
        line_y = full_cy - int(p * pixels_per_degree) - pitch_shift_y
        if line_y < 0: continue # Não desenha fora da tela
        
        line_length = 60 if abs(p) % 20 == 0 else 30 # Mais longo para múltiplos de 20
        cv2.line(overlay, (full_cx - line_length, line_y), (full_cx + line_length, line_y), LINE_COLOR, 1)
        cv2.putText(overlay, str(p), (full_cx + line_length + 5, line_y + 5), FONT, 0.6, LINE_COLOR, 1)

    # Linhas de -10, -20, -30, etc. (abaixo do horizonte)
    for p in range(-10, -91, -10): # Linhas negativas
        line_y = full_cy - int(p * pixels_per_degree) - pitch_shift_y
        if line_y > canvas.shape[0]: continue # Não desenha fora da tela
        
        line_length = 60 if abs(p) % 20 == 0 else 30
        cv2.line(overlay, (full_cx - line_length, line_y), (full_cx + line_length, line_y), LINE_COLOR, 1)
        cv2.putText(overlay, str(p), (full_cx + line_length + 5, line_y + 5), FONT, 0.6, LINE_COLOR, 1)


    # --- Rotacionar o Overlay (para o Roll) ---
    M = cv2.getRotationMatrix2D((full_cx, full_cy), -roll_deg, 1)
    rotated_overlay = cv2.warpAffine(overlay, M, (canvas.shape[1], canvas.shape[0]))
    
    # --- Mesclar o Overlay Rotacionado com o Canvas Principal ---
    # Usamos o cv2.addWeighted para mesclar de forma transparente (ou cv2.add para simples overlay)
    # Para um OSD, geralmente cv2.add é suficiente, pois as linhas são finas
    canvas[:] = cv2.add(canvas, rotated_overlay) # Adiciona as linhas sobre o vídeo


    # --- Desenhar o Símbolo Fixo da Aeronave (no centro da tela) ---
    # Estes símbolos não rotacionam com o roll/pitch, são fixos na tela
    symbol_arm_length = 50
    symbol_gap = 10
    
    # Asa Esquerda
    cv2.line(canvas, (full_cx - symbol_arm_length, full_cy), (full_cx - symbol_gap, full_cy), LINE_COLOR, 3)
    # Asa Direita
    cv2.line(canvas, (full_cx + symbol_gap, full_cy), (full_cx + symbol_arm_length, full_cy), LINE_COLOR, 3)
    # Linha central vertical (indicador de nariz)
    cv2.line(canvas, (full_cx, full_cy - symbol_gap), (full_cx, full_cy + symbol_gap), LINE_COLOR, 3)
    
    # --- Desenhar o Indicador de Roll (Linha horizontal no topo da tela) ---
    # Isso pode ser feito com uma pequena escala de roll no topo, se desejar
    # Por enquanto, vamos fazer um indicador simples
    roll_indicator_y = 10 # Posição vertical para o indicador no topo
    roll_indicator_len = 40 # Comprimento do indicador
    
    # Desenha um pequeno "triângulo" que indica o roll
    roll_line_start = (full_cx + int(math.sin(roll) * roll_indicator_len), 
                       roll_indicator_y + int(math.cos(roll) * roll_indicator_len / 2))
    roll_line_end = (full_cx - int(math.sin(roll) * roll_indicator_len), 
                     roll_indicator_y - int(math.cos(roll) * roll_indicator_len / 2))
    #cv2.line(canvas, roll_line_start, roll_line_end, LINE_COLOR, 2)
    
    # Uma forma mais simples: uma pequena seta no topo
    roll_arrow_x = full_cx + int(math.sin(roll) * (WIDTH//2 - 20)) # Move ao longo do topo
    cv2.line(canvas, (roll_arrow_x, roll_indicator_y), (roll_arrow_x, roll_indicator_y + 10), LINE_COLOR, 2)
    # Adicionar as marcas de roll no topo (opcional, pode ficar poluído)
    # for r in range(-60, 61, 30):
    #     mark_x = full_cx + int(math.sin(math.radians(r)) * (WIDTH//2 - 40))
    #     cv2.line(canvas, (mark_x, roll_indicator_y), (mark_x, roll_indicator_y + 5), LINE_COLOR, 1)

def draw_tape(canvas, value, x_pos, y_pos, width, height, is_vertical=True, color=(0, 255, 0), tick_range=50, step=10):
    """Desenha uma fita de altitude ou velocidade com fundo translúcido."""
    center_y = y_pos + height // 2
    center_x = x_pos + width // 2
    
    # --- NOVO: Fundo de Vidro (Transparente) ---
    # 1. Definir a transparência (alpha). 0.3 = 30% opaco.
    alpha = 0.3
    beta = 1.0 - alpha
    
    # 2. Extrair a Região de Interesse (ROI) do canvas
    # Garantir que não saia dos limites da tela
    y1, y2 = max(0, y_pos), min(canvas.shape[0], y_pos + height)
    x1, x2 = max(0, x_pos), min(canvas.shape[1], x_pos + width)
    
    if y1 < y2 and x1 < x2: # Se a ROI for válida
        roi_tape = canvas[y1:y2, x1:x2]
        
        # 3. Criar o overlay preto (mesmo tamanho da ROI)
        black_overlay = np.zeros_like(roi_tape)
        
        # 4. Misturar a ROI com o overlay
        blended_roi = cv2.addWeighted(roi_tape, beta, black_overlay, alpha, 0)
        
        # 5. Colocar a ROI misturada de volta no canvas
        canvas[y1:y2, x1:x2] = blended_roi
    # --- FIM DA MUDANÇA NO FUNDO ---
    
    # Borda da fita (continua igual)
    cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + width, y_pos + height), color, 1)

    # Definir uma opacidade maior para o fundo do texto (para legibilidade)
    alpha_text = 0.5
    beta_text = 1.0 - alpha_text

    if is_vertical:
        # --- NOVO: Fundo de Vidro para o Marcador Central ---
        y1_val, y2_val = center_y - 15, center_y + 15
        x1_val, x2_val = x_pos, x_pos + width + 20
        
        # Garantir limites
        y1_val, y2_val = max(0, y1_val), min(canvas.shape[0], y2_val)
        x1_val, x2_val = max(0, x1_val), min(canvas.shape[1], x2_val)

        if y1_val < y2_val and x1_val < x2_val:
            roi_val = canvas[y1_val:y2_val, x1_val:x2_val]
            black_overlay_val = np.zeros_like(roi_val)
            blended_val = cv2.addWeighted(roi_val, beta_text, black_overlay_val, alpha_text, 0)
            canvas[y1_val:y2_val, x1_val:x2_val] = blended_val
        # --- FIM DA MUDANÇA ---
        
        # Texto do marcador (continua igual)
        cv2.putText(canvas, f"{int(value):>3}", (x_pos + 5, center_y + 10), FONT, 0.8, color, 2)
        
        # Desenhar marcas da fita (continua igual)
        pixels_per_unit = height / tick_range
        int_val = int(value)
        
        for i in range(int_val - tick_range, int_val + tick_range):
            if i % step == 0:
                y = center_y - int((i - value) * pixels_per_unit)
                if y > y_pos and y < y_pos + height:
                    cv2.line(canvas, (x_pos + width - 20, y), (x_pos + width, y), color, 2)
                    cv2.putText(canvas, str(i), (x_pos + 5, y + 5), FONT, 0.5, color, 1)

    else: # Fita Horizontal (Bússola)
        # --- NOVO: Fundo de Vidro para o Marcador Central ---
        y1_comp, y2_comp = y_pos - 30, y_pos
        x1_comp, x2_comp = center_x - 20, center_x + 20

        y1_comp, y2_comp = max(0, y1_comp), min(canvas.shape[0], y2_comp)
        x1_comp, x2_comp = max(0, x1_comp), min(canvas.shape[1], x2_comp)

        if y1_comp < y2_comp and x1_comp < x2_comp:
            roi_comp = canvas[y1_comp:y2_comp, x1_comp:x2_comp]
            black_overlay_comp = np.zeros_like(roi_comp)
            blended_comp = cv2.addWeighted(roi_comp, beta_text, black_overlay_comp, alpha_text, 0)
            canvas[y1_comp:y2_comp, x1_comp:x2_comp] = blended_comp
        # --- FIM DA MUDANÇA ---
        
        # Texto do marcador (continua igual)
        cv2.putText(canvas, f"{int(value):03}", (center_x - 18, y_pos - 8), FONT, 0.8, color, 2)
        cv2.line(canvas, (center_x, y_pos), (center_x, y_pos + 10), color, 2)
        
        # Marcas da fita (continua igual)
        pixels_per_unit = width / tick_range
        int_val = int(value)
        
        for i in range(int_val - tick_range, int_val + tick_range):
            if i % 10 == 0: # Marca a cada 10 graus
                x = center_x - int((i - value) * pixels_per_unit)
                if x > x_pos and x < x_pos + width:
                    i_norm = i % 360 # Normalizar 0-360
                    lbl = str(i_norm)
                    if i_norm == 0: lbl = "N"
                    elif i_norm == 90: lbl = "E"
                    elif i_norm == 180: lbl = "S"
                    elif i_norm == 270: lbl = "W"
                        
                    cv2.line(canvas, (x, y_pos), (x, y_pos + 10), color, 2)
                    if i_norm % 30 == 0: # Rótulos maiores
                        cv2.putText(canvas, lbl, (x - 10, y_pos + 30), FONT, 0.6, color, 1)


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
                    export_model = YOLO(str(source))
                    export_model.export(format=YOLO_EXPORT_FORMAT, imgsz=YOLO_IMGSZ)
                    exported_source = backend_sources[YOLO_EXPORT_FORMAT]
                    if exported_source.exists():
                        selected_backend = YOLO_EXPORT_FORMAT
                        selected_source = exported_source
                except Exception as exc:
                    print(f"Aviso: falha ao exportar YOLO para {YOLO_EXPORT_FORMAT} ({exc}); seguindo com PT.")

            YOLO_ACTIVE_BACKEND = selected_backend
            print(f"YOLO backend ativo: {YOLO_ACTIVE_BACKEND} ({selected_source})")
            return YOLO(str(selected_source))

        if source.exists():
            if source.suffix.lower() == ".onnx":
                YOLO_ACTIVE_BACKEND = "onnx"
            elif source.is_dir() and source.name.endswith("_openvino_model"):
                YOLO_ACTIVE_BACKEND = "openvino"
            else:
                YOLO_ACTIVE_BACKEND = "pt"

            print(f"YOLO backend ativo: {YOLO_ACTIVE_BACKEND} ({source})")
            return YOLO(str(source))

        YOLO_ACTIVE_BACKEND = "pt"
        print(f"YOLO backend ativo: {YOLO_ACTIVE_BACKEND} ({YOLO_MODEL_SOURCE})")
        return YOLO(YOLO_MODEL_SOURCE)
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
            detections.append({
                "index": index,
                "bbox": (x1, y1, x2, y2),
                "center": (person_x, person_y),
                "offset": (offset_x, offset_y),
            })

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
        cv2.putText(frame, label, (x1, label_y), FONT, 0.5, OSD_COLOR, 2)


def save_person_snapshot(scene, detections, capture_dir=CAPTURE_DIR):
    if not detections:
        return None

    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = capture_dir / f"person_{timestamp}.jpg"

    grayscale_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(
        str(filename),
        grayscale_scene,
        [cv2.IMWRITE_JPEG_QUALITY, 60],
    )
    return filename


def draw_status_banner(canvas, text, color=OSD_COLOR):
    banner_width = min(canvas.shape[1] - 40, max(280, len(text) * 12 + 40))
    banner_height = 44
    x1 = (canvas.shape[1] - banner_width) // 2
    y1 = 20
    x2 = x1 + banner_width
    y2 = y1 + banner_height

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, text, (x1 + 18, y1 + 29), FONT, 0.8, color, 2)


def configure_fullscreen_window(window_name, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Loop Principal da Interface ---

def main():
    global sim_data, WIDTH, HEIGHT # Usar o dicionário global e a resolução detectada
    WIDTH, HEIGHT = get_display_resolution()
    person_detector = load_person_detector()
    person_detection_enabled = False
    person_capture_enabled = False
    status_text = ""
    status_until = 0.0
    cam2, cam2_source = open_camera(2, "/dev/v4l/by-id/*USB_CAM2*")
    if not cam2:
        print(f"Erro: Nao foi possivel abrir a camera {cam2_source}")
        print("Usando simulacao de fallback.")
        cam2 = None # Define como None se falhar

    am1, am1_source = open_camera(0, "/dev/v4l/by-id/*USB_CAM1*")
    if not am1:
        print(f"Erro: Nao foi possivel abrir a camera {am1_source}")
        print("Usando simulacao de fallback.")
        am1 = None
    # Estado da Câmera
    thermal_is_main = sim_data["thermal_is_main"]
    window_name = "FPV Interface Sim"
    last_telemetry_update = 0.0
    last_display_frame_time = 0.0
    stop_event = threading.Event()
    state_lock = threading.Lock()

    normal_render_queue = Queue(maxsize=1)
    thermal_render_queue = Queue(maxsize=1)
    normal_infer_queue = Queue(maxsize=1)
    thermal_infer_queue = Queue(maxsize=1)

    normal_fallback = create_error_frame((480, 640, 3), "CAMERA 0 ERROR", (80, 40, 40))
    thermal_fallback = create_error_frame((192, 256, 3), "CAMERA 2 ERROR")
    last_normal_frame = normal_fallback.copy()
    last_thermal_frame = thermal_fallback.copy()

    shared_state = {
        "person_detection_enabled": False,
        "normal_detections": [],
        "thermal_detections": [],
    }

    worker_threads = []
    if am1:
        normal_capture_thread = threading.Thread(
            target=capture_worker,
            args=(am1, [normal_render_queue, normal_infer_queue], stop_event),
            daemon=True,
        )
        normal_capture_thread.start()
        worker_threads.append(normal_capture_thread)

    if cam2:
        thermal_capture_thread = threading.Thread(
            target=capture_worker,
            args=(cam2, [thermal_render_queue, thermal_infer_queue], stop_event),
            daemon=True,
        )
        thermal_capture_thread.start()
        worker_threads.append(thermal_capture_thread)

    yolo_thread = threading.Thread(
        target=yolo_inference_worker,
        args=(normal_infer_queue, thermal_infer_queue, person_detector, shared_state, state_lock, stop_event),
        daemon=True,
    )
    yolo_thread.start()
    worker_threads.append(yolo_thread)

    configure_fullscreen_window(window_name, WIDTH, HEIGHT)

    last_time = time.time()

    try:
        while True:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            if current_time - last_telemetry_update >= 1.0 / TELEMETRY_HZ:
                t = current_time * 0.5
                sim_data["roll"] = math.sin(t * 0.7) * 30
                sim_data["pitch"] = math.cos(t * 0.5) * 15
                sim_data["heading"] = (sim_data["heading"] + delta_time * 5) % 360
                sim_data["altitude"] = 100 + (math.sin(t * 0.2) * 20)
                sim_data["airspeed"] = 20 + (math.sin(t * 0.3) * 5)
                sim_data["ground_speed"] = sim_data["airspeed"] - 1.5
                sim_data["sats"] = 12 + int(math.sin(t))
                sim_data["batt_volt"] -= delta_time * 0.01
                sim_data["lon"] += delta_time * 0.0001
                last_telemetry_update = current_time

            dist_m = abs(sim_data["lon"] - sim_data["home_lon"]) * 111111

            last_normal_frame = get_latest_or_last(normal_render_queue, last_normal_frame)
            last_thermal_frame = get_latest_or_last(thermal_render_queue, last_thermal_frame)

            frame_normal = last_normal_frame.copy()
            frame_thermal = last_thermal_frame.copy()

            with state_lock:
                normal_detections = list(shared_state["normal_detections"])
                thermal_detections = list(shared_state["thermal_detections"])

            if person_detection_enabled:
                draw_person_detections(frame_normal, normal_detections)
                draw_person_detections(frame_thermal, thermal_detections)

            if thermal_is_main:
                main_frame = frame_thermal
                pip_frame = frame_normal
                pip_frame_resized = cv2.resize(pip_frame, (178, 133))
            else:
                main_frame = frame_normal
                pip_frame = frame_thermal
                pip_frame_resized = cv2.resize(pip_frame, (160, 120))

            scene = cv2.resize(main_frame, (WIDTH, HEIGHT))
            pip_h, pip_w = pip_frame_resized.shape[:2]
            scene[HEIGHT-pip_h-10 : HEIGHT-10, WIDTH-pip_w-10 : WIDTH-10] = pip_frame_resized
            cv2.rectangle(scene, (WIDTH-pip_w-10, HEIGHT-pip_h-10), (WIDTH-10, HEIGHT-10), OSD_COLOR, 1)

            draw_artificial_horizon(scene, sim_data["roll"], sim_data["pitch"],
                                    cx=WIDTH // 2, cy=HEIGHT // 2 - 50, radius=100)

            draw_tape(scene, sim_data["airspeed"], x_pos=40, y_pos=100,
                      width=70, height=HEIGHT - 200, is_vertical=True, color=OSD_COLOR, tick_range=20, step=5)
            cv2.putText(scene, "IAS", (45, 90), FONT, 0.7, OSD_COLOR, 1)

            draw_tape(scene, sim_data["altitude"], x_pos=WIDTH - 110, y_pos=100,
                      width=70, height=HEIGHT - 200, is_vertical=True, color=OSD_COLOR, tick_range=50, step=10)
            cv2.putText(scene, "ALT", (WIDTH - 105, 90), FONT, 0.7, OSD_COLOR, 1)

            draw_tape(scene, sim_data["heading"], x_pos=150, y_pos=50,
                      width=WIDTH - 300, height=30, is_vertical=False, color=OSD_COLOR, tick_range=60, step=10)

            cv2.putText(scene, f"M: {sim_data['flight_mode']}", (15, 30), FONT, 0.7, OSD_COLOR, 1)
            cv2.putText(scene, f"GPS: {sim_data['sats']} SAT", (15, 60), FONT, 0.5, OSD_COLOR, 1)

            if person_detection_enabled:
                cv2.putText(scene, "DET PESSOAS: ON", (15, 90), FONT, 0.6, OSD_COLOR, 2)
            if person_capture_enabled:
                cv2.putText(scene, "PRINT YOLO: ON", (15, 120), FONT, 0.6, OSD_COLOR, 2)

            cv2.putText(scene, f"{sim_data['batt_volt']:.1f}V", (WIDTH - 100, 30), FONT, 0.7, OSD_COLOR, 1)
            cv2.putText(scene, f"LAT {sim_data['lat']:.5f}", (15, HEIGHT - 40), FONT, 0.6, OSD_COLOR, 1)
            cv2.putText(scene, f"LON {sim_data['lon']:.5f}", (15, HEIGHT - 15), FONT, 0.6, OSD_COLOR, 1)
            cv2.putText(scene, f"H {int(dist_m)}m", (WIDTH//2 - 40, HEIGHT - 15), FONT, 0.7, OSD_COLOR, 2)

            active_detections = normal_detections + thermal_detections
            if person_capture_enabled and active_detections:
                snapshot_path = save_person_snapshot(scene, active_detections)
                if snapshot_path is not None:
                    status_text = f"PRINT SALVO: {snapshot_path.name}"
                    status_until = current_time + 2.0

            if current_time < status_until and status_text:
                draw_status_banner(scene, status_text)

            if current_time - last_display_frame_time >= 1.0 / DISPLAY_FPS:
                cv2.imshow(window_name, scene)
                last_display_frame_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                thermal_is_main = not thermal_is_main
                sim_data["thermal_is_main"] = thermal_is_main
            if key == ord('d'):
                person_detection_enabled = not person_detection_enabled
                with state_lock:
                    shared_state["person_detection_enabled"] = person_detection_enabled
                    if not person_detection_enabled:
                        shared_state["normal_detections"] = []
                        shared_state["thermal_detections"] = []
                status_text = (
                    "DETECCAO DE PESSOAS ATIVADA" if person_detection_enabled else "DETECCAO DE PESSOAS DESATIVADA"
                )
                status_until = current_time + 2.0
            if key == ord('c'):
                person_capture_enabled = not person_capture_enabled
                status_text = (
                    "PRINT YOLO ATIVADO" if person_capture_enabled else "PRINT YOLO DESATIVADO"
                )
                status_until = current_time + 2.0
    finally:
        stop_event.set()
        for thread in worker_threads:
            thread.join(timeout=0.5)

        if cam2:
            cam2.release()
        if am1:
            am1.release()
        cv2.destroyAllWindows()

# --- Rodar o Programa ---
if __name__ == "__main__":
    main()