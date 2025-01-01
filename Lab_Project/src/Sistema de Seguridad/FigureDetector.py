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

    def detect_shape(self, frame):
        """Detects the figure in the given frame.

        Args:
            frame (numpy.ndarray): Frame from the camera in which the figure will be searched.

        Returns:
            bool: True if the figure is detected, False otherwise.
        """
        # Arreglar distorsión
        # frame = self.undistort_frame(frame)

        # la gausiiana
        # blur = cv2.GaussianBlur(frame, (5, 5), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,self.figure.lower_color,self.figure.upper_color)
        cv2.imshow('mask', mask)

        img_masked = cv2.bitwise_and(frame,frame,mask=mask)
        

        cv2.imshow("result", img_masked)

        # # Convert to grayscale and apply threshold
        # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # #Binariza para el erode de despues con adaptativos para que no dependa de la iluminación

        # Apply erosion to reduce noise
        kernel = np.ones((5, 5), np.uint8) #Dimensiones del kernel 
        eroded = cv2.erode(mask, kernel, iterations=2)

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
                # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                # cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                # cv2.imshow("Mask",np.asarray(mask,dtype=np.float64))
                # mean_color = cv2.mean(frame, mask=mask)[:3]
                # print(mean_color)
                # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                # cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                # mean_color_bgr = cv2.mean(frame, mask=mask)[:3]
                
                # # Convertir BGR a RGB
                # mean_color_rgb = (mean_color_bgr[2], mean_color_bgr[1], mean_color_bgr[0])
            
                # #Calcula el promedio de los valores de píxeles en una imagen (frame) que están dentro de una región especificada por una máscara binaria (mask).
                # #Devuelve una tupla de 4 elementos: (mean_blue, mean_green, mean_red, mean_alpha). Xeso nos quedamos con [:3]
                # if self.figure.color_within_tolerance(mean_color_rgb):
                    # return True
                return True
        return False


    def draw_detected_shape(self, frame): #De chat
        """Draws on the frame.

        Args:
            frame (numpy.ndarray): Frame from the camera where the figure will be drawn.
        """
        # Draw the figure in the top-left corner
        x, y, size = 20, 20, 20  # Top-left corner and size of the shape
        cv2.rectangle(frame, (x, y), (x + size, y + size), self.figure.get_color_rgb(), thickness=-1)
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
    valid_figures = [
        Figure("circle", "blue", (0,112, 192), np.array([65,120,69]), np.array([91,255,98]), 9), #cuantas esquinas le pongo?
        Figure("square", "green", (0,176,80), np.array([40, 100, 50]), np.array([100, 255, 255]), 4), # (0,176,80)
        Figure("triangle", "green", (255,0,0), np.array([65,120,69]), np.array([91,255,98]), 3),
        Figure("pentagon", "purple", (112,48,160), np.array([65,120,69]), np.array([91,255,98]), 5),
    ]

    # Intentamos solo el cuadrado, luego probar el circulo 
    square = valid_figures[0]

    detector = FigureDetector(square)
    
    while True:
        frame = detector.picam.capture_array() # no se muy bien como captura esto los frames, de chat
        detected = detector.detect_shape(frame)

        if detected:
            detector.draw_detected_shape(frame)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()