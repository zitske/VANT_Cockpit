import cv2
import math
import time
import struct
import serial
import numpy as np

# Configura a porta serial (substitua 'COM3' pela porta correta no seu sistema)
# No Mac, a porta geralmente é algo como '/dev/tty.usbmodem14101'
ser = serial.Serial('/dev/cu.usbserial-1130', 115200, timeout=5)
# Verifica se a porta serial foi aberta corretamente
if not ser.is_open:
    print("Erro ao abrir a porta serial")
    exit()

# Definição da estrutura de dados recebida do Arduino
class SensorData:
    def __init__(self, pitch, roll, temperature, pressure, altitude, airspeed, airTemperature, airpressure, magX, magY, magZ):
        self.pitch = pitch
        self.roll = roll
        self.temperature = temperature
        self.pressure = pressure
        self.altitude = altitude
        self.airspeed = airspeed
        self.airTemperature = airTemperature
        self.airpressure = airpressure
        self.magX = magX
        self.magY = magY
        self.magZ = magZ

# Variáveis globais para armazenar os últimos valores válidos
last_valid_data = SensorData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Função para ler dados da porta serial
def read_serial_data():
    global last_valid_data
    if ser.in_waiting >= 44:  # Tamanho da estrutura SensorData em bytes
        data = ser.read(44)
        unpacked_data = struct.unpack('11f', data)
        sensor_data = SensorData(*unpacked_data)
        
        # Verifica se algum valor é NaN e substitui pelo último valor válido
        for attr in vars(sensor_data):
            value = getattr(sensor_data, attr)
            if np.isnan(value):
                setattr(sensor_data, attr, getattr(last_valid_data, attr))
            else:
                setattr(last_valid_data, attr, value)
        
        return sensor_data
    return None

# Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(1)

# Verifica se a webcam foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

# Adiciona um pequeno atraso para garantir que a câmera esteja pronta
time.sleep(2)

while True:
    sensor_data = read_serial_data()
    if sensor_data:
        print(f"Dados recebidos: {sensor_data.__dict__}")

    # Captura frame por frame
    ret, frame = cap.read()

    # Se o frame foi capturado corretamente
    if not ret:
        print("Erro ao capturar o frame")
        break

    # Recebe os dados do sensor
    pitch = sensor_data.pitch if sensor_data else last_valid_data.pitch
    roll = sensor_data.roll if sensor_data else last_valid_data.roll
    air_speed = sensor_data.airspeed if sensor_data else last_valid_data.airspeed
    altura = sensor_data.altitude if sensor_data else last_valid_data.altitude
    angle = 0
    #sensor_data.airpressure if sensor_data else last_valid_data.airpressure
    direction = "N"  # Direção fictícia, pode ser calculada com base em outros dados

    # Adiciona um overlay (por exemplo, um círculo vermelho no centro da tela)
    overlay = frame.copy()

    # Desenha o horizonte artificial
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    horizon_length = 200
    horizon_x1 = int(center_x - horizon_length * math.cos(math.radians(roll)))
    horizon_y1 = int(center_y - horizon_length * math.sin(math.radians(roll)))
    horizon_x2 = int(center_x + horizon_length * math.cos(math.radians(roll)))
    horizon_y2 = int(center_y + horizon_length * math.sin(math.radians(roll)))

    cv2.line(overlay, (horizon_x1, horizon_y1), (horizon_x2, horizon_y2), (0, 255, 0), 2)

    # Defina a opacidade desejada (0.0 a 1.0)
    opacity = 0.5
    
    # Crie uma cópia do overlay original
    overlay_copy = overlay.copy()
    
    # Desenha os mostradores de velocidade na cópia do overlay
    speed_display_width = 150
    speed_display_height = height - 100
    speed_display_y = 50
    speed_left_x = 50
    
    # Mostrador de velocidade esquerdo
    cv2.rectangle(overlay_copy, 
                  (speed_left_x, speed_display_y), 
                  (speed_left_x + speed_display_width, speed_display_y + speed_display_height), 
                  (255, 255, 255), -1)
    cv2.putText(overlay_copy, f"{round(air_speed*1.94384)} Kn", 
                (speed_left_x + speed_display_width//4,  (speed_display_height//2)+speed_display_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Mostrador de altitude direito
    altitude_display_y = speed_display_y
    altitude_right_x = width - speed_display_width - speed_display_y
    altitude_display_width = speed_display_width
    cv2.rectangle(overlay_copy, (altitude_right_x, altitude_display_y), 
                  (altitude_right_x + altitude_display_width, altitude_display_y + speed_display_height), 
                  (255, 255, 255), -1)
    cv2.putText(overlay_copy, f"{round(altura*3.28084)} Ft", 
                (altitude_right_x + altitude_display_width//4, speed_display_y + (speed_display_height//2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Mostrador de direção
    direction_display_height = 100
    direction_display_x = 250
    direction_display_width = width - direction_display_x
    top_padding = 25
    cv2.rectangle(overlay_copy, 
                  (direction_display_x, top_padding), 
                  (direction_display_width, direction_display_height), 
                  (255, 255, 255), -1)
    cv2.putText(overlay_copy, f"{angle}{direction}", 
                (direction_display_width//2+direction_display_x//4, direction_display_height-top_padding), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Combine o overlay com a imagem original usando a opacidade
    cv2.addWeighted(overlay_copy, opacity, overlay, 1 - opacity, 0, overlay)

    alpha = 1  # Transparência do overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Mostra o frame com o overlay
    cv2.imshow('Webcam com Overlay', frame)

    # Sai do loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()