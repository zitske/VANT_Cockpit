import cv2
import numpy as np
import math
import time

# --- Configurações da Interface ---
WIDTH, HEIGHT = 720, 576 # Resolução PAL SD
FONT = cv2.FONT_HERSHEY_SIMPLEX
OSD_COLOR = (0, 255, 0) # Verde clássico de FPV

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

# --- Funções de Desenho da Interface ---

def create_simulated_frames(t):
    """Gera dois frames de câmera falsos e animados."""
    
    # Frame 1 (Simulando Câmera Normal)
    frame_normal = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_normal[:, :] = (80, 40, 40) # Fundo azul escuro
    cv2.putText(frame_normal, "NORMAL CAM", (180, 240), FONT, 1, (255, 255, 255), 2)
    # Adiciona um "ruído" animado
    noise = np.random.randint(0, 15, (480, 640, 3), dtype=np.uint8)
    frame_normal = cv2.add(frame_normal, noise)

    # Frame 2 (Simulando Câmera Térmica)
    frame_thermal = np.zeros((192, 256, 3), dtype=np.uint8)
    frame_thermal[:, :] = (50, 50, 50) # Fundo cinza
    cv2.putText(frame_thermal, "THERMAL", (60, 100), FONT, 0.8, (255, 255, 255), 1)
    # Simula um "hotspot" térmico
    cx = int(128 + math.sin(t) * 50)
    cy = int(96 + math.cos(t * 0.7) * 30)
    cv2.circle(frame_thermal, (cx, cy), 20, (0, 165, 255), -1) # Círculo laranja
    frame_thermal = cv2.applyColorMap(frame_thermal, cv2.COLORMAP_INFERNO)
    
    return frame_normal, frame_thermal

def draw_artificial_horizon(canvas, roll_deg, pitch_deg, cx, cy, radius):
    """Desenha um Horizonte Artificial (AHI) completo."""
    
    # Cores
    SKY_COLOR = (255, 128, 0) # Azul
    GROUND_COLOR = (0, 80, 130) # Marrom

    # Converter para radianos
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    
    # --- 1. Desenhar o Fundo (Céu e Terra) ---
    # Criar um canvas temporário para o mundo rotacionado
    world_size = radius * 3
    world_canvas = np.zeros((world_size, world_size, 3), dtype=np.uint8)
    
    # Calcular o deslocamento do pitch (quantos pixels o horizonte se move)
    # 'pixels_per_degree' é um fator de zoom; 5 é um bom valor
    pixels_per_degree = 5
    pitch_shift = int(pitch_deg * pixels_per_degree)
    
    # Calcular o ponto central do horizonte
    horizon_y = (world_size // 2) - pitch_shift
    
    # Desenhar Céu e Terra
    cv2.rectangle(world_canvas, (0, 0), (world_size, horizon_y), SKY_COLOR, -1)
    cv2.rectangle(world_canvas, (0, horizon_y), (world_size, world_size), GROUND_COLOR, -1)

    # --- 2. Desenhar a Escala de Pitch ---
    for p in range(-90, 91, 10): # Linhas a cada 10 graus
        if p == 0: continue
        line_y = (world_size // 2) - int(p * pixels_per_degree) - pitch_shift
        
        if abs(p) % 20 == 0: # Linha mais longa
            cv2.line(world_canvas, (world_size//2 - 60, line_y), (world_size//2 + 60, line_y), (255, 255, 255), 2)
            cv2.putText(world_canvas, str(p), (world_size//2 - 90, line_y + 5), FONT, 0.6, (255, 255, 255), 1)
        else: # Linha mais curta
            cv2.line(world_canvas, (world_size//2 - 30, line_y), (world_size//2 + 30, line_y), (255, 255, 255), 1)

    # --- 3. Rotacionar o Mundo (para o Roll) ---
    M = cv2.getRotationMatrix2D((world_size // 2, world_size // 2), -roll_deg, 1)
    rotated_world = cv2.warpAffine(world_canvas, M, (world_size, world_size))
    
    # --- 4. Recortar e Colar no Canvas Principal ---
    # Pegar o centro do mundo rotacionado e colar no canvas principal
    x_start = (world_size // 2) - (radius)
    x_end = (world_size // 2) + (radius)
    y_start = (world_size // 2) - (radius)
    y_end = (world_size // 2) + (radius)
    
    # Criar uma máscara circular
    mask = np.zeros((radius*2, radius*2, 3), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)
    
    # Aplicar máscara e copiar
    roi = canvas[cy-radius:cy+radius, cx-radius:cx+radius]
    display_world = rotated_world[y_start:y_end, x_start:x_end]
    display_world_masked = cv2.bitwise_and(display_world, mask)
    
    # Criar um fundo preto para a máscara
    bg_black = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    
    canvas[cy-radius:cy+radius, cx-radius:cx+radius] = cv2.add(bg_black, display_world_masked)

    # --- 5. Desenhar o Símbolo Fixo da Aeronave ---
    cv2.line(canvas, (cx - 40, cy), (cx - 10, cy), OSD_COLOR, 3)
    cv2.line(canvas, (cx + 10, cy), (cx + 40, cy), OSD_COLOR, 3)
    cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), OSD_COLOR, 3) # V-bar central
    
    # --- 6. Desenhar o Indicador de Roll (Topo) ---
    cv2.line(canvas, (cx, cy - radius), (cx, cy - radius + 15), OSD_COLOR, 3)
    # Desenha o triângulo de roll
    r_x1 = cx + int((radius - 10) * math.sin(roll))
    r_y1 = cy - int((radius - 10) * math.cos(roll))
    cv2.line(canvas, (cx, cy), (r_x1, r_y1), OSD_COLOR, 2)

def draw_tape(canvas, value, x_pos, y_pos, width, height, is_vertical=True, color=(0, 255, 0), tick_range=50, step=10):
    """Desenha uma fita de altitude ou velocidade."""
    center_y = y_pos + height // 2
    center_x = x_pos + width // 2
    
    # Fundo da fita
    cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + width, y_pos + height), color, 1)

    if is_vertical:
        # Marcador central (valor)
        cv2.rectangle(canvas, (x_pos, center_y - 15), (x_pos + width + 20, center_y + 15), (0,0,0), -1)
        cv2.putText(canvas, f"{int(value):>3}", (x_pos + 5, center_y + 10), FONT, 0.8, color, 2)
        
        # Desenhar marcas da fita
        pixels_per_unit = height / tick_range
        int_val = int(value)
        
        for i in range(int_val - tick_range, int_val + tick_range):
            if i % step == 0:
                y = center_y - int((i - value) * pixels_per_unit)
                if y > y_pos and y < y_pos + height:
                    cv2.line(canvas, (x_pos + width - 20, y), (x_pos + width, y), color, 2)
                    cv2.putText(canvas, str(i), (x_pos + 5, y + 5), FONT, 0.5, color, 1)

    else: # Fita Horizontal (Bússola)
        # Marcador central (valor)
        cv2.rectangle(canvas, (center_x - 20, y_pos - 30), (center_x + 20, y_pos), (0,0,0), -1)
        cv2.putText(canvas, f"{int(value):03}", (center_x - 18, y_pos - 8), FONT, 0.8, color, 2)
        cv2.line(canvas, (center_x, y_pos), (center_x, y_pos + 10), color, 2)
        
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


# --- Loop Principal da Interface ---

def main():
    global sim_data # Usar o dicionário global

    # Estado da Câmera
    thermal_is_main = sim_data["thermal_is_main"]

    last_time = time.time()

    while True:
        # --- 1. Atualizar Dados Simulados ---
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        
        t = current_time * 0.5 # Fator de tempo para animação
        
        # Simular voo
        sim_data["roll"] = math.sin(t * 0.7) * 30  # +/- 30 graus de roll
        sim_data["pitch"] = math.cos(t * 0.5) * 15 # +/- 15 graus de pitch
        sim_data["heading"] = (sim_data["heading"] + delta_time * 5) % 360 # Girando 5 deg/s
        sim_data["altitude"] = 100 + (math.sin(t * 0.2) * 20) # Variando entre 80-120m
        sim_data["airspeed"] = 20 + (math.sin(t * 0.3) * 5) # Variando entre 15-25 m/s
        sim_data["ground_speed"] = sim_data["airspeed"] - 1.5 # Vento leve
        sim_data["sats"] = 12 + int(math.sin(t))
        sim_data["batt_volt"] -= delta_time * 0.01 # Bateria descarregando lentamente
        sim_data["lon"] += delta_time * 0.0001 # Movendo para o leste
        
        # Calcular distância de casa (simplificado)
        dist_m = abs(sim_data["lon"] - sim_data["home_lon"]) * 111111 # Aproximação
        
        # --- 2. Criar o Canvas Principal ---
        # Fundo transparente (preto) sobre o qual desenhamos
        scene = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # --- 3. Desenhar Feeds de Câmera ---
        frame_normal, frame_thermal = create_simulated_frames(current_time)
        
        # Atribuir com base no estado de troca
        if thermal_is_main:
            main_frame = frame_thermal
            pip_frame = frame_normal
            # Redimensionar para caber
            main_frame_resized = cv2.resize(main_frame, (512, 384)) # 256*2, 192*2
            pip_frame_resized = cv2.resize(pip_frame, (178, 133)) # PiP
        else:
            main_frame = frame_normal
            pip_frame = frame_thermal
            main_frame_resized = cv2.resize(main_frame, (640, 480))
            pip_frame_resized = cv2.resize(pip_frame, (160, 120)) # PiP

        # Colocar o frame principal (centralizado no topo)
        main_h, main_w = main_frame_resized.shape[:2]
        main_x = (WIDTH - main_w) // 2
        main_y = (HEIGHT - main_h) // 2 - 50 # Um pouco para cima
        if main_y < 0: 
            main_y = 0
        scene[main_y:main_y+main_h, main_x:main_x+main_w] = main_frame_resized

        # Colocar o PiP (canto inferior direito)
        pip_h, pip_w = pip_frame_resized.shape[:2]
        scene[HEIGHT-pip_h-10 : HEIGHT-10, WIDTH-pip_w-10 : WIDTH-10] = pip_frame_resized
        # Borda no PiP
        cv2.rectangle(scene, (WIDTH-pip_w-10, HEIGHT-pip_h-10), (WIDTH-10, HEIGHT-10), OSD_COLOR, 1)

        # --- 4. Desenhar OSD e PFD ---
        
        # Horizonte Artificial (centralizado)
        draw_artificial_horizon(scene, sim_data["roll"], sim_data["pitch"], 
                                cx=WIDTH // 2, cy=HEIGHT // 2 - 50, radius=100)

        # Fita de Velocidade (Airspeed) - Esquerda
        draw_tape(scene, sim_data["airspeed"], x_pos=40, y_pos=100, 
                  width=70, height=HEIGHT - 200, is_vertical=True, color=OSD_COLOR, tick_range=20, step=5)
        cv2.putText(scene, "IAS", (45, 90), FONT, 0.7, OSD_COLOR, 1)
        
        # Fita de Altitude - Direita
        draw_tape(scene, sim_data["altitude"], x_pos=WIDTH - 110, y_pos=100, 
                  width=70, height=HEIGHT - 200, is_vertical=True, color=OSD_COLOR, tick_range=50, step=10)
        cv2.putText(scene, "ALT", (WIDTH - 105, 90), FONT, 0.7, OSD_COLOR, 1)

        # Fita da Bússola - Topo
        draw_tape(scene, sim_data["heading"], x_pos=150, y_pos=50, 
                  width=WIDTH - 300, height=30, is_vertical=False, color=OSD_COLOR, tick_range=60, step=10)
        
        # --- 5. Desenhar Textos do OSD ---
        # Canto Superior Esquerdo
        cv2.putText(scene, f"M: {sim_data['flight_mode']}", (15, 30), FONT, 0.7, OSD_COLOR, 1)
        cv2.putText(scene, f"GPS: {sim_data['sats']} SAT", (15, 60), FONT, 0.7, OSD_COLOR, 1)

        # Canto Superior Direito
        cv2.putText(scene, f"{sim_data['batt_volt']:.1f}V", (WIDTH - 100, 30), FONT, 0.7, OSD_COLOR, 1)

        # Canto Inferior Esquerdo
        cv2.putText(scene, f"LAT {sim_data['lat']:.5f}", (15, HEIGHT - 40), FONT, 0.6, OSD_COLOR, 1)
        cv2.putText(scene, f"LON {sim_data['lon']:.5f}", (15, HEIGHT - 15), FONT, 0.6, OSD_COLOR, 1)
        
        # Canto Inferior (Centro) - Home
        cv2.putText(scene, f"H {int(dist_m)}m", (WIDTH//2 - 40, HEIGHT - 15), FONT, 0.7, OSD_COLOR, 2)


        # --- 6. Exibir a Cena ---
        # 'scene' é o frame final que será enviado para a saída AV
        cv2.imshow("FPV Interface Sim", scene)

        # --- 7. Lidar com Entradas ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # 'q' para Sair
            break
        if key == ord('s'): # 's' para Trocar (Swap)
            thermal_is_main = not thermal_is_main
            sim_data["thermal_is_main"] = thermal_is_main # Atualiza o estado global


    cv2.destroyAllWindows()

# --- Rodar o Programa ---
if __name__ == "__main__":
    main()