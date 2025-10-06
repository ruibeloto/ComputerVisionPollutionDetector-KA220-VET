from picamera2 import Picamera2
from time import sleep
from datetime import datetime
import os

def capture_pollution_image(filename_prefix="pollution"):

    # Criar diretório se não existir
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Gerar nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"images/{filename_prefix}_{timestamp}.jpg"
    
    # Capturar imagem
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    sleep(2)  # Auto-ajuste
    picam2.capture_file(filename)
    picam2.stop()
    
    return filename


image_path = capture_pollution_image()
print(f"Imagem capturada: {image_path}")
