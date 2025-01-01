import cv2
from picamera2 import Picamera2
import numpy as np
from Figure import Figure

class FigureDetector:
    def __init__(self):
        # Initiate camera (de chat)
        self.picam = Picamera2()
        self.picam.preview_configuration.main.size = (640, 480)
        self.picam.preview_configuration.main.format = "RGB888"  #entiendo que esto está bien? HARIA FALTA CAMBIAR LOS FAMRES DE COLOR?
        self.picam.preview_configuration.align()
        self.picam.configure("preview")
        self.picam.start()

        #Figura a la que vamos a ver si encuentra en un determinado frame
        self.figure = None

    def set_figure(self, figure: Figure):
        """
        Sets the figure that is going to be detected.

        Args:
            figure (Figure): figure to be set in the detector.

        Returns: None
        """
        self.figure = figure

    def detect_shape(self, frame):
        """Detects the figure in the given frame.

        Args:
            frame (numpy.ndarray): Frame from the camera in which the figure will be searched.

        Returns:
            bool: True if the figure is detected, False otherwise.
        """

        # la gausiiana
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,self.figure.lower_color,self.figure.upper_color)

        # Apply erosion to reduce noise
        kernel = np.ones((5, 5), np.uint8) #Dimensiones del kernel 
        eroded = cv2.erode(mask, kernel, iterations=6)
        
        # Find contours in the frame
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the polygon
            corners = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            if self.figure.figure_type == "circle": 
                # Check if the contour resembles a circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:  # Avoid division by zero
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.8 <= circularity <= 1.2:  # Circularity threshold for circles
                    return True

            elif len(corners) == self.figure.n_vertex:
                return True
        return False

    def draw_detected_shape(self, frame):
        """
        Draws the detected figure in the top-left corner of the frame.
        
        Args:
            frame (numpy.ndarray): Frame where the figure will be drawn
        """
        x, y = 20, 20  # Top-left corner position
        size = 30  # Size of the shape
        color = self.figure.get_color_bgr()
        
        if self.figure.figure_type == "square":
            cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
        
        elif self.figure.figure_type == "triangle":
            points = np.array([
                [x + size//2, y],
                [x, y + size],
                [x + size, y + size]
            ], dtype=np.int32)
            cv2.fillPoly(frame, [points], color)
        
        elif self.figure.figure_type == "circle":
            center = (x + size//2, y + size//2)
            radius = size//2
            cv2.circle(frame, center, radius, color, -1)
        
        elif self.figure.figure_type == "pentagon":
            radius = size//2
            center = (x + size//2, y + size//2)
            points = []
            for i in range(5):
                angle = i * 2 * np.pi / 5 - np.pi/2  # Starting from top
                point_x = center[0] + int(radius * np.cos(angle))
                point_y = center[1] + int(radius * np.sin(angle))
                points.append([point_x, point_y])
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(frame, [points], color)

        # Add figure type label
        cv2.putText(
            frame,
            self.figure.figure_type,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

if __name__ == "__main__":
    valid_figures = [
        Figure("circle", "blue", (0,112, 192), np.array([100,130,0]), np.array([179,255,255]), 9), 
        Figure("square", "green", (0,176,80), np.array([40, 100, 50]), np.array([100, 255, 255]), 4), # (0,176,80)
        Figure("triangle", "red", (255,0,0), np.array([0,155,113]), np.array([7,210,155]), 3),
        Figure("pentagon", "purple", (112,48,160), np.array([123,125,23]), np.array([161,255,178]), 5),
    ]

    # Intentamos solo el cuadrado, luego probar el circulo 
    # square = valid_figures[0]

    detector = FigureDetector()
    
    while True:
        frame = detector.picam.capture_array() # no se muy bien como captura esto los frames, de chat
  
        for figure in valid_figures:
            detector.set_figure(figure)
            detected = detector.detect_shape(frame)

            if detected:
                detector.draw_detected_shape(frame)
                break

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()