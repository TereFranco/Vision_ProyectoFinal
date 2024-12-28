import cv2
import time
from FigureDetector import FigureDetector
from Figure import Figure

def check_password(password):
    detected_pattern = []
    #A VER SI PODEMOS ESPERAR PARA HACER LAS DETECCIONES
    #SI NO FUNCIONA PODEMOS VER EL CODIGO DE SERGIO
    last_detection_time = 0
    detection_delay = 2  

    while True:
        current_time = time.time()
        for figure in valid_figures:
            if current_time - last_detection_time >= detection_delay: #Si ha pasado el tiempo suficiente
                frame = detector.picam.capture_array()#picam activada del FigureDetector, pillamos el frame ahora
                #Detectamos
                detected = detector.detect_shape(frame, figure)
                if detected: #si detectamos guardamos la figura
                    detected_pattern.append(figure.figure_type)
                    last_detection_time = current_time
                    detector.draw_detected_shape(frame, figure.figure_type) #debería verse en la pantalla
                    print(f"Se ha detectado un: {figure.figure_type}") #No tenemos más información de lo que se haya mostrado
                    break
    
        #Ahora para gstionar la contraseña
        if len(detected_pattern) == 4:
            if detected_pattern == password:
                print("El patrón coincide. Ahora se iniciará el tracker")
                break
            else:
                print("El patrón no coincide, vuélvalo a intentar.")
                detected_pattern = []  # Resetemos no? o lo podemos hacer de otra forma nose

        # Para ver las figuras que se han detectado, de chat
        cv2.putText(
            frame,
            f"Pattern: {' -> '.join(detected_pattern)}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FigureDetector()

    valid_figures = [
        Figure("circle", "blue", (0, 112, 192), 9),
        Figure("square", "green", (0, 176, 80), 4),
        Figure("triangle", "green", (255, 0, 0), 3),
        Figure("pentagon", "purple", (112, 48, 160), 5),
    ]

    #our password
    password = ["circle", "square", "triangle", "pentagon"]  
    check_password(password)