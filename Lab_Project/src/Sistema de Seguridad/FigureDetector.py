import cv2
from picamera2 import Picamera2
import numpy as np
from Figure import Figure

class FigureDetector:
    def __init__(self, figure):
        # Initiate camera (de chat)
        self.picam = Picamera2()
        self.picam.preview_configuration.main.size = (640, 480)
        self.picam.preview_configuration.main.format = "RGB888"  #entiendo que esto está bien? HARIA FALTA CAMBIAR LOS FAMRES DE COLOR?
        self.picam.preview_configuration.align()
        self.picam.configure("preview")
        self.picam.start()

        #Figura a la que vamos a ver si encuentra en un determinado frame
        self.figure = figure

        # Cargar los parámetros de calibración
        calibration_data = np.load('Calibration/calibration_data.npz') #No se si funciona este path porque está en una carpeta de fuera!!
        self.mtx = calibration_data['intrinsics']
        self.dist = calibration_data['dist_coeffs']

    def undistort_frame(self, frame):
        """Fixes the distortion of the frame.

        Args:
            frame (numpy.ndarray): Frame from the camera that needs to be corrected.

        Returns:
            numpy.ndarray: Corrected frame without distortion.
        """
        h, w = frame.shape[:2]
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        return cv2.undistort(frame, self.mtx, self.dist, None, new_mtx)


    def detect_shape(self, frame):
        """Detects the figure in the given frame.

        Args:
            frame (numpy.ndarray): Frame from the camera in which the figure will be searched.

        Returns:
            bool: True if the figure is detected, False otherwise.
        """
        # Arreglar distorsión
        frame = self.undistort_frame(frame)

        # la gausiiana
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #Binariza para el erode de despues con adaptativos para que no dependa de la iluminación

        # Apply erosion to reduce noise
        kernel = np.ones((5, 5), np.uint8) #Dimensiones del kernel 
        eroded = cv2.erode(thresh, kernel, iterations=2)

        cv2.imshow("eroded", eroded) #TERE usa esto para ver si esta pillandolo
        
        # Find contours in the frame
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the polygon
            corners = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            #TODO DE CHAT PARA IDENTIFICAR EL CIRCULO
            if self.figure.figure_type == "circle": 
                # Check if the contour resembles a circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:  # Avoid division by zero
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.8 <= circularity <= 1.2:  # Circularity threshold for circles
                    # Create a mask and calculate the mean color within the contour
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                    mean_color = cv2.mean(frame, mask=mask)[:3]

                    if self.figure.color_within_tolerance(mean_color):
                        return True

            elif len(corners) == self.figure.n_vertex:
                # Create a mask and calculate the mean color within the contour
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                mean_color = cv2.mean(frame, mask=mask)[:3]

                #Calcula el promedio de los valores de píxeles en una imagen (frame) que están dentro de una región especificada por una máscara binaria (mask).
                #Devuelve una tupla de 4 elementos: (mean_blue, mean_green, mean_red, mean_alpha). Xeso nos quedamos con [:3]
                if self.figure.color_within_tolerance(mean_color):
                    return True
        return False


    def draw_detected_shape(self, frame): #De chat
        """Draws on the frame.

        Args:
            frame (numpy.ndarray): Frame from the camera where the figure will be drawn.
        """
        # Draw the figure in the top-left corner
        x, y, size = 20, 20, 100  # Top-left corner and size of the shape
        cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 2)
        cv2.putText(
            frame,
            self.figure.figure_type,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


if __name__ == "__main__":
    detector = FigureDetector()

    valid_figures = [
        Figure("circle", "blue", (0,112, 192), 9), #cuantas esquinas le pongo?
        Figure("square", "green", (0,176,80), 4),
        Figure("triangle", "green", (255,0,0), 3),
        Figure("pentagon", "purple", (112,48,160), 5),
        
    ]

    # Intentamos solo el cuadrado, luego probar el circulo 
    square = valid_figures[1]
    while True:
        frame = detector.picam.capture_array() # no se muy bien como captura esto los frames, de chat
        detected = detector.detect_shape(frame, square)
        if detected:
            detector.draw_detected_shape(frame, square.figure_type)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
