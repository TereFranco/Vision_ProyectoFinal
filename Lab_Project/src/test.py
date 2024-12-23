import cv2
from picamera2 import Picamera2
import os

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    i = 0
    
    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        
        # Capturar la tecla presionada
        key = cv2.waitKey(1) & 0xFF
        
        # Verificar si se presionó 'q' para salir
        if key == ord('q'):
            break
            
        # Verificar si se presionó 'f' para capturar
        elif key == ord('f'):
            filename = f"pattern_{i}.jpg"
            filepath = os.path.join("./patterns", filename)
            image_captured = cv2.imwrite(filepath, frame)
            if image_captured:
                print(f"Imagen capturada y guardada como '{filename}'")
                i += 1
    
    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs("./patterns", exist_ok=True)
    stream_video()

