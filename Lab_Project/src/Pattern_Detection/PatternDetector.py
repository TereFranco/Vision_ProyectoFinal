import cv2
from picamera2 import Picamera2
import numpy as np
from enum import auto, Enum
import time

class PatternStates(Enum):
    CIRCLE = auto()
    SQUARE = auto()
    CIRCLE_W_LINE = auto()
    SQUARE_W_LINE = auto()

class PatternDetector:
    def __init__(self):
        self.picam = Picamera2()
        self.picam.preview_configuration.main.size=(640, 480)
        self.picam.preview_configuration.main.format="RGB888"
        self.picam.preview_configuration.align()
        self.picam.configure("preview")
        self.picam.start()
        self.pattern_state = PatternStates.CIRCLE
        self.detected_patterns = []
        self.unlock_key = [
            PatternStates.CIRCLE,
            PatternStates.SQUARE,
            PatternStates.CIRCLE_W_LINE,
            PatternStates.SQUARE_W_LINE
        ]
        self.unlocked = False

        # Variables for draw_patterns() function
        self.margin = 20  # Margen superior
        self.size = 30  # Tamaño de cada figura
        self.spacing = 20  # Espaciado entre figuras
        self.patterns_thickness = 2
        # Coordenadas iniciales para dibujar
        self.x_offset = self.spacing
        self.y_offset = self.margin

        # Para segmentación de color
        self.lower_white = np.array([72,0,149])
        self.upper_white = np.array([255,255,255])

        self.lower_black = np.array([74, 15, 117])
        self.upper_black = np.array([117, 59, 161])

        self.last_detection_time = time.time()
        # Cargar los parámetros de calibración
        calibration_data = np.load('Calibration/calibration_data.npz') #No sé si funciona el directorio 
        self.mtx = calibration_data['intrinsics']
        self.dist = calibration_data['dist_coeffs']

    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        return cv2.undistort(frame, self.mtx, self.dist, None, new_mtx)
    
    def detect_shapes(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        mask = cv2.bitwise_or(black_mask, white_mask)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles, squares, lines = [], [], []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(corners) > 8:  # Posible círculo
                (x, y), radius = cv2.minEnclosingCircle(contour)
                area_circle = np.pi * radius**2
                area_contour = cv2.contourArea(contour)
                if 0.8 < area_contour / area_circle < 1.2:
                    circles.append(contour)
            elif len(corners) == 4:  # Posible cuadrado
                (x, y, w, h) = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:
                    squares.append(contour)
            else:  # Posible línea
                if len(contour) >= 5:  # Se necesita al menos 5 puntos para fitLine
                    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    length = cv2.arcLength(contour, True)
                    if length > 50:
                        lines.append((x, y, vx, vy))

        return circles, squares, lines

    
    def check_pattern(self, frame):
        circles, squares, lines = self.detect_shapes(frame)
        
        # Dibujar formas detectadas
        cv2.drawContours(frame, circles, -1, (0, 255, 0), 2) # en verde
        cv2.drawContours(frame, squares, -1, (255, 0, 0), 2) # en azul
        
        if lines:
            for x, y, vx, vy in lines:
                cv2.line(frame, (int(x-vx*100), int(y-vy*100)), (int(x+vx*100), int(y+vy*100)), (0,0,255), 2)
            
        # Detectar y registrar formas
        if circles and not lines:
            self.detected_patterns.append(PatternStates.CIRCLE)
        elif squares and not lines:
            self.detected_patterns.append(PatternStates.SQUARE)
        elif circles and lines:
            if any(self.line_intersects_shape(line, circle) for line in lines for circle in circles):
                self.detected_patterns.append(PatternStates.CIRCLE_W_LINE)
        elif squares and lines:
            if any(self.line_intersects_shape(line, square) for line in lines for square in squares):
                self.detected_patterns.append(PatternStates.SQUARE_W_LINE)

        # Verificar si la lista tiene 4 elementos
        if len(self.detected_patterns) == 4:
            if self.detected_patterns == self.unlock_key:
                self.unlocked = True
            else:
                print("Clave incorrecta. Reinicia la lista para volver a intentarlo.")
                self.detected_patterns = []  # Reiniciar la lista automáticamente

        # for i, pattern in enumerate(self.detected_patterns):
        #     cv2.putText(frame, f"{pattern.name}", (10, 30 + i*30),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        return frame
    
    def line_intersects_shape(self, line, shape):
        if line is None or shape is None:
            return False
        x, y, vx, vy = line
        
        # Verificar varios puntos a lo largo de la línea
        for t in range(-100, 101, 5):
            point = (int(x + vx*t), int(y + vy*t))
            if cv2.pointPolygonTest(shape, point, False) >= 0:
                return True
        return False
    
    def draw_patterns(self,frame):
        x_offset = self.x_offset
        
        for pattern in self.detected_patterns:
            if pattern == PatternStates.CIRCLE:
                # Dibujar un círculo grueso
                center = (x_offset + self.size // 2, self.y_offset + self.size // 2)
                cv2.circle(frame, center, self.size // 2, (0,0,0), thickness=self.patterns_thickness)
            elif pattern == PatternStates.SQUARE:
                # Dibujar un cuadrado grueso
                top_left = (x_offset, self.y_offset)
                bottom_right = (x_offset + self.size, self.y_offset + self.size)
                cv2.rectangle(frame, top_left, bottom_right, (0,0,0), thickness=self.patterns_thickness)
            elif pattern == PatternStates.CIRCLE_W_LINE:
                # Dibujar un círculo con una línea
                center = (x_offset + self.size // 2, self.y_offset + self.size // 2)
                cv2.circle(frame, center, self.size // 2, (0,0,0), thickness=self.patterns_thickness)
                cv2.line(frame, (center[0], center[1] - self.size // 2),
                        (center[0], center[1] + self.size // 2), (0,0,0), thickness=self.patterns_thickness)
            elif pattern == PatternStates.SQUARE_W_LINE:
                # Dibujar un cuadrado con una línea
                top_left = (x_offset, self.y_offset)
                bottom_right = (x_offset + self.size, self.y_offset + self.size)
                cv2.rectangle(frame, top_left, bottom_right, (0,0,0), thickness=self.patterns_thickness)
                cv2.line(frame, (x_offset + self.size // 2, self.y_offset),
                        (x_offset + self.size // 2, self.y_offset + self.size), (0,0,0), thickness=self.patterns_thickness)
            
            # Avanzar para el siguiente dibujo
            x_offset += self.size + self.spacing

        return frame

    def run(self):
        while True:
            current_time = time.time()
            frame = self.picam.capture_array()
            # frame = self.undistort_frame(frame)

            # Solo procesar cada 100ms para mejorar el rendimiento
            if current_time - self.last_detection_time > 0.1:
                if not self.unlocked:
                    frame = self.check_pattern(frame)
                else:
                    print("Sistema desbloqueado - Iniciando tracking...")
                    break

            self.last_detection_time = current_time
            # Mostrar patrones detectados
            frame = self.draw_patterns(frame)
            cv2.imshow("picam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        return self.unlocked

if __name__ == "__main__":
    detector = PatternDetector()
    unlocked = detector.run()
    if unlocked:
        print("Sistema desbloqueado - Iniciando tracking...")