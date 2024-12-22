import cv2
from picamera2 import Picamera2
from figure_detection_functions import *

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    first_pattern_sequence = []
    second_pattern_sequence = []
    third_pattern_sequence = []
    fourth_pattern_sequence = []
    
    identification_completed = False

    while True:
        frame = picam.capture_array()
        
        # Capturar la tecla presionada
        key = cv2.waitKey(1) & 0xFF
        
        # Verificar si se presionó 'q' para salir
        if key == ord('q'):
            break

        if not identification_completed:
            if len(first_pattern_sequence) < len(FIRST_PATTERN_SEQUENCE):
                frame, last_pattern_detected = figure_detection(frame, first_pattern_sequence, last_pattern_detected)
                if len(first_pattern_sequence) == len(FIRST_PATTERN_SEQUENCE) and first_pattern_sequence != FIRST_PATTERN_SEQUENCE:
                    first_pattern_sequence = []
            elif len(second_pattern_sequence) < len(SECOND_PATTERN_SEQUENCE):
                frame, last_pattern_detected = figure_detection(frame, second_pattern_sequence, last_pattern_detected)
                if len(second_pattern_sequence) == len(SECOND_PATTERN_SEQUENCE) and second_pattern_sequence != SECOND_PATTERN_SEQUENCE:
                    second_pattern_sequence = []
            elif len(third_pattern_sequence) < len(THIRD_PATTERN_SEQUENCE):
                frame, last_pattern_detected = figure_detection(frame, third_pattern_sequence, last_pattern_detected)
                if len(third_pattern_sequence) == len(THIRD_PATTERN_SEQUENCE) and third_pattern_sequence != THIRD_PATTERN_SEQUENCE:
                    third_pattern_sequence = False
            elif len(fourth_pattern_sequence) < len(FOURTH_PATTERN_SEQUENCE):
                frame, last_pattern_detected = figure_detection(frame, fourth_pattern_sequence, last_pattern_detected)
                if len(fourth_pattern_sequence) == len(FOURTH_PATTERN_SEQUENCE) and fourth_pattern_sequence != FOURTH_PATTERN_SEQUENCE:
                    fourth_pattern_sequence = False
            else:
                identification_completed = True

        cv2.imshow("picam", frame)
        
    
    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()