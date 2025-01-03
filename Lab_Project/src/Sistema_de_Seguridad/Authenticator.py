import cv2
import time
from .FigureDetector import FigureDetector
from .Figure import Figure
import numpy as np
from .initialize_picam import initialize
from picamera2 import Picamera2

class Authenticator:
    def __init__(self, picam: Picamera2, password: list):
        self.figure_detector: FigureDetector = FigureDetector(picam)
        self.password: list[Figure] = password
        self.detected_pattern: list[Figure] = []
        self.valid_figures: list[Figure] = [
            Figure("circle", "blue", (0,112, 192), np.array([100,130,0]), np.array([179,255,255]), 9), 
            Figure("square", "green", (0,176,80), np.array([40, 100, 50]), np.array([100, 255, 255]), 4),
            Figure("triangle", "red", (255,0,0), np.array([0,155,113]), np.array([7,210,155]), 3),
            Figure("pentagon", "purple", (112,48,160), np.array([123,125,23]), np.array([161,255,178]), 5)
        ]
    

    def draw_rectangle(self, frame: np.ndarray, color: tuple) -> np.ndarray:
        """
        Draws a rectangle on the frame with the specified color (green if the password is correct, red otherwise).

        Args:
            frame (numpy.ndarray): The image frame on which the rectangle will be drawn.
            color (tuple): The color of the rectangle in (B, G, R) format.

        Returns:
            frame (np.ndarray): The image frame with the rectangle drawn.
        """
        cv2.rectangle(frame, (0, 0), (640, 480), color, 10)
        return frame

    def check_password(self):
        """
        Checks if the detected pattern matches the given password.

        Args: None

        Returns: None
        """
        last_detection_time = 0
        detection_delay = 2
        show_result = False
        result_start_time = 0
        password_matched = False
        prev_time = 0

        while True:
            frame = self.figure_detector.picam.capture_array()
            current_time = time.time()

            # Calcular FPS
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            fps = int(fps)
            fps_text = f"{fps} FPS"
            # Muestra los FPS en la esquina superior derecha del vídeo
            height, width, _ = frame.shape
            cv2.putText(frame, fps_text, (width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            if show_result:
                if current_time - result_start_time >= detection_delay:
                    if password_matched:
                        break
                    show_result = False
                    self.detected_pattern = []
                else:
                    if password_matched:
                        frame = self.draw_rectangle(frame, (0, 255, 0))
                    else:
                        frame = self.draw_rectangle(frame, (0, 0, 255))

            if not show_result:
                for figure in self.valid_figures:
                    if current_time - last_detection_time >= detection_delay:
                        self.figure_detector.set_figure(figure)
                        detected = self.figure_detector.detect_shape(frame)
                        if detected:
                            self.detected_pattern.append(figure)
                            last_detection_time = current_time
                            break

            frame = self.draw_detected_pattern(frame)
            
            if len(self.detected_pattern) == 4 and not show_result:
                show_result = True
                result_start_time = current_time
                if self.detected_pattern == self.password:
                    print("El patrón coincide.")
                    frame = self.draw_rectangle(frame, (0, 255, 0))
                    password_matched = True
                else:
                    print("El patrón no coincide.")
                    frame = self.draw_rectangle(frame, (0, 0, 255))

            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()

    def draw_detected_pattern(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the figures of the detected pattern filled with their color on the frame.

        Args:
            frame (np.ndarray): frame on which detected figures will be drawn.

        Returns: 
            frame (np.ndarray): frame with detected figures drawn.
        """
        spacing = 10  # Space between figures
        size = 30     # Figure's size
        base_x = 20   # Initial X position
        y = 20        # Fixed Y position

        for i, figure in enumerate(self.detected_pattern):
            x = base_x + (i * (size + spacing))  # Calculates X coordinate for each detected pattern
            color = figure.get_color_bgr()
            
            if figure.get_figure_type() == "square":
                cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
            
            elif figure.get_figure_type() == "triangle":
                points = np.array([
                    [x + size//2, y],
                    [x, y + size],
                    [x + size, y + size]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [points], color)
            
            elif figure.get_figure_type() == "circle":
                center = (x + size//2, y + size//2)
                radius = size//2
                cv2.circle(frame, center, radius, color, -1)
            
            elif figure.get_figure_type() == "pentagon":
                radius = size//1.75
                center = (x + size//2, y + size//2)
                points = [] # list with the points defining the pentagon
                for i in range(5):
                    angle = i * 2 * np.pi / 5 - np.pi/2
                    point_x = center[0] + int(radius * np.cos(angle))
                    point_y = center[1] + int(radius * np.sin(angle))
                    points.append([point_x, point_y])
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(frame, [points], color)

        return frame

if __name__ == "__main__":
    #our password
    password = [
        Figure("circle", "blue", (0,112, 192), np.array([100,130,0]), np.array([179,255,255]), 9), 
        Figure("square", "green", (0,176,80), np.array([40, 100, 50]), np.array([100, 255, 255]), 4),
        Figure("triangle", "red", (255,0,0), np.array([0,155,113]), np.array([7,210,155]), 3),
        Figure("pentagon", "purple", (112,48,160), np.array([123,125,23]), np.array([161,255,178]), 5)
    ]

    picam = initialize()
    authenticator = Authenticator(picam, password)
    authenticator.check_password()
    picam.stop()