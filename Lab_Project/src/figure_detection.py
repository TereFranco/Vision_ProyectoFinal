import cv2
from picamera2 import Picamera2
from figure_detection_functions import *
import time

def stream_video():
    # Inicializar la cámara con una resolución más baja para mejor rendimiento
    picam = Picamera2()
    picam.preview_configuration.main.size=(640, 480)  # Reducida de 1280x720
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    # Tiempo de espera para estabilizar la cámara
    time.sleep(2)
    
    pattern_sequences = {
        'first': {'sequence': [], 'target': FIRST_PATTERN_SEQUENCE},
        'second': {'sequence': [], 'target': SECOND_PATTERN_SEQUENCE},
        'third': {'sequence': [], 'target': THIRD_PATTERN_SEQUENCE},
        'fourth': {'sequence': [], 'target': FOURTH_PATTERN_SEQUENCE}
    }
    
    identification_completed = False
    last_pattern_detected = None
    last_detection_time = time.time()
    
    while True:
        frame = picam.capture_array()
        
        current_time = time.time()
        # Solo procesar cada 100ms para mejorar el rendimiento
        if current_time - last_detection_time > 0.1:
            if not identification_completed:
                for seq_name, seq_data in pattern_sequences.items():
                    if len(seq_data['sequence']) < len(seq_data['target']):
                        frame, detected_pattern = figure_detection(
                            frame, 
                            seq_data['sequence'], 
                            last_pattern_detected,
                            min_confidence=0.85  # Aumentado el umbral de confianza
                        )
                        last_pattern_detected = detected_pattern
                        
                        # Si la secuencia está completa pero es incorrecta, reiniciarla
                        if (len(seq_data['sequence']) == len(seq_data['target']) and 
                            seq_data['sequence'] != seq_data['target']):
                            seq_data['sequence'] = []
                        break
                else:
                    identification_completed = True
            
            last_detection_time = current_time
            
        # Mostrar las secuencias detectadas
        y_pos = 30
        for seq_name, seq_data in pattern_sequences.items():
            cv2.putText(frame, 
                       f"{seq_name}: {seq_data['sequence']}", 
                       (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 255, 255), 
                       1)
            y_pos += 20
        
        cv2.imshow("Pattern Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()