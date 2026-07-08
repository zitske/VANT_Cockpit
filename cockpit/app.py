import threading
import time
from queue import Queue

import cv2

from .camera import capture_worker, create_error_frame, get_latest_or_last, open_camera
from .config import DETECTION_HZ, DISPLAY_FPS, FONT, HEIGHT, OSD_COLOR, TELEMETRY_HZ, WIDTH, has_gui_display
from .osd import draw_artificial_horizon, draw_status_banner, draw_tape
from .simulation import sim_data, update_simulation
from .yolo import detect_persons, draw_person_detections, load_person_detector, save_person_snapshot


def configure_fullscreen_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def set_window_fullscreen(window_name, width, height, fullscreen_enabled):
    if fullscreen_enabled:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, 0, 0)


def run():
    person_detector = load_person_detector()
    person_detection_enabled = False
    person_capture_enabled = False
    nav_hud_enabled = False
    status_text = ""
    status_until = 0.0

    cam2, cam2_source = open_camera(2, "/dev/v4l/by-id/*USB_CAM2*")
    if not cam2:
        print(f"Erro: Nao foi possivel abrir a camera {cam2_source}")
        print("Usando simulacao de fallback.")
        cam2 = None

    am1, am1_source = open_camera(0, "/dev/v4l/by-id/*USB_CAM1*")
    if not am1:
        print(f"Erro: Nao foi possivel abrir a camera {am1_source}")
        print("Usando simulacao de fallback.")
        am1 = None

    thermal_is_main = sim_data["thermal_is_main"]
    window_name = "FPV Interface Sim"
    last_telemetry_update = 0.0
    last_display_frame_time = 0.0
    stop_event = threading.Event()
    state_lock = threading.Lock()
    gui_enabled = has_gui_display()

    normal_render_queue = Queue(maxsize=1)
    thermal_render_queue = Queue(maxsize=1)

    normal_fallback = create_error_frame((480, 640, 3), "CAMERA 0 ERROR", (80, 40, 40))
    thermal_fallback = create_error_frame((192, 256, 3), "CAMERA 2 ERROR")
    last_normal_frame = normal_fallback.copy()
    last_thermal_frame = thermal_fallback.copy()

    shared_state = {
        "person_detection_enabled": False,
        "active_detections": [],
    }

    worker_threads = []
    if am1:
        normal_capture_thread = threading.Thread(
            target=capture_worker,
            args=(am1, [normal_render_queue], stop_event),
            daemon=True,
        )
        normal_capture_thread.start()
        worker_threads.append(normal_capture_thread)

    if cam2:
        thermal_capture_thread = threading.Thread(
            target=capture_worker,
            args=(cam2, [thermal_render_queue], stop_event),
            daemon=True,
        )
        thermal_capture_thread.start()
        worker_threads.append(thermal_capture_thread)

    if gui_enabled:
        try:
            configure_fullscreen_window(window_name)
        except cv2.error as exc:
            gui_enabled = False
            print(f"Aviso: interface gráfica indisponível, executando em modo headless: {exc}")
    else:
        print("Aviso: nenhuma sessão gráfica detectada, executando em modo headless.")

    last_time = time.time()
    last_detection_time = 0.0
    active_detections = []
    fullscreen_enabled = True

    try:
        while True:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            if current_time - last_telemetry_update >= 1.0 / TELEMETRY_HZ:
                update_simulation(sim_data, current_time, delta_time)
                last_telemetry_update = current_time

            dist_m = abs(sim_data["lon"] - sim_data["home_lon"]) * 111111

            last_normal_frame = get_latest_or_last(normal_render_queue, last_normal_frame)
            last_thermal_frame = get_latest_or_last(thermal_render_queue, last_thermal_frame)

            frame_normal = last_normal_frame.copy()
            frame_thermal = last_thermal_frame.copy()

            if thermal_is_main:
                main_frame = frame_thermal
                pip_frame = frame_normal
                pip_frame_resized = cv2.resize(pip_frame, (178, 133))
            else:
                main_frame = frame_normal
                pip_frame = frame_thermal
                pip_frame_resized = cv2.resize(pip_frame, (160, 120))

            if not person_detection_enabled:
                active_detections = []
            elif person_detector is not None and current_time - last_detection_time >= 1.0 / DETECTION_HZ:
                active_detections = detect_persons(main_frame, person_detector)
                last_detection_time = current_time

            if person_detection_enabled and active_detections:
                draw_person_detections(main_frame, active_detections)

            with state_lock:
                shared_state["active_detections"] = list(active_detections)

            scene = cv2.resize(main_frame, (WIDTH, HEIGHT))
            pip_h, pip_w = pip_frame_resized.shape[:2]
            scene[HEIGHT - pip_h - 10 : HEIGHT - 10, WIDTH - pip_w - 10 : WIDTH - 10] = pip_frame_resized
            cv2.rectangle(scene, (WIDTH - pip_w - 10, HEIGHT - pip_h - 10), (WIDTH - 10, HEIGHT - 10), OSD_COLOR, 1)

            if nav_hud_enabled:
                draw_artificial_horizon(scene, sim_data["roll"], sim_data["pitch"], cx=WIDTH // 2, cy=HEIGHT // 2 - 50, radius=100)
                draw_tape(scene, sim_data["airspeed"], x_pos=40, y_pos=100, width=70, height=HEIGHT - 200, is_vertical=True, color=OSD_COLOR, tick_range=20, step=5)
                cv2.putText(scene, "IAS", (45, 90), FONT, 0.7, OSD_COLOR, 1)
                draw_tape(scene, sim_data["altitude"], x_pos=WIDTH - 110, y_pos=100, width=70, height=HEIGHT - 200, is_vertical=True, color=OSD_COLOR, tick_range=50, step=10)
                cv2.putText(scene, "ALT", (WIDTH - 105, 90), FONT, 0.7, OSD_COLOR, 1)
                draw_tape(scene, sim_data["heading"], x_pos=150, y_pos=50, width=WIDTH - 300, height=30, is_vertical=False, color=OSD_COLOR, tick_range=60, step=10)
                cv2.putText(scene, f"M: {sim_data['flight_mode']}", (15, 30), FONT, 0.7, OSD_COLOR, 1)
                cv2.putText(scene, f"GPS: {sim_data['sats']} SAT", (15, 60), FONT, 0.5, OSD_COLOR, 1)
                if person_detection_enabled:
                    cv2.putText(scene, "DET PESSOAS: ON", (15, 90), FONT, 0.6, OSD_COLOR, 2)
                if person_capture_enabled:
                    cv2.putText(scene, "PRINT YOLO: ON", (15, 120), FONT, 0.6, OSD_COLOR, 2)
                cv2.putText(scene, f"{sim_data['batt_volt']:.1f}V", (WIDTH - 100, 30), FONT, 0.7, OSD_COLOR, 1)
                cv2.putText(scene, f"LAT {sim_data['lat']:.5f}", (15, HEIGHT - 40), FONT, 0.6, OSD_COLOR, 1)
                cv2.putText(scene, f"LON {sim_data['lon']:.5f}", (15, HEIGHT - 15), FONT, 0.6, OSD_COLOR, 1)
                cv2.putText(scene, f"H {int(dist_m)}m", (WIDTH // 2 - 40, HEIGHT - 15), FONT, 0.7, OSD_COLOR, 2)

            if person_capture_enabled and active_detections:
                snapshot_path = save_person_snapshot(scene, active_detections)
                if snapshot_path is not None:
                    status_text = f"PRINT SALVO: {snapshot_path.name}"
                    status_until = current_time + 2.0

            if current_time < status_until and status_text:
                draw_status_banner(scene, status_text)

            if gui_enabled and current_time - last_display_frame_time >= 1.0 / DISPLAY_FPS:
                try:
                    cv2.imshow(window_name, scene)
                    last_display_frame_time = current_time
                except cv2.error as exc:
                    gui_enabled = False
                    print(f"Aviso: falha ao exibir janela OpenCV, mudando para modo headless: {exc}")

            key = cv2.waitKey(1) & 0xFF if gui_enabled else 255
            if key == ord("q"):
                break
            if key == ord("s"):
                thermal_is_main = not thermal_is_main
                sim_data["thermal_is_main"] = thermal_is_main
                last_detection_time = 0.0
                active_detections = []
            if key == ord("h"):
                nav_hud_enabled = not nav_hud_enabled
                status_text = "HUD DE NAVEGACAO ATIVADO" if nav_hud_enabled else "HUD DE NAVEGACAO DESATIVADO"
                status_until = current_time + 2.0
            if key == ord("f"):
                fullscreen_enabled = not fullscreen_enabled
                if gui_enabled:
                    set_window_fullscreen(window_name, WIDTH, HEIGHT, fullscreen_enabled)
                status_text = "FULLSCREEN ATIVADO" if fullscreen_enabled else "JANELA NORMAL ATIVADA"
                status_until = current_time + 2.0
            if key == ord("d"):
                person_detection_enabled = not person_detection_enabled
                with state_lock:
                    shared_state["person_detection_enabled"] = person_detection_enabled
                    if not person_detection_enabled:
                        shared_state["active_detections"] = []
                        active_detections = []
                status_text = "DETECCAO DE PESSOAS ATIVADA" if person_detection_enabled else "DETECCAO DE PESSOAS DESATIVADA"
                status_until = current_time + 2.0
            if key == ord("c"):
                person_capture_enabled = not person_capture_enabled
                status_text = "PRINT YOLO ATIVADO" if person_capture_enabled else "PRINT YOLO DESATIVADO"
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
