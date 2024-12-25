import cv2
from picamera2 import Picamera2
import numpy as np
from enum import auto, Enum

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

        # Cargar los parámetros de calibración
        # calibration_data = np.load('calibration_data.npz')
        # self.mtx = calibration_data['mtx']
        # self.dist = calibration_data['dist']

    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        return cv2.undistort(frame, self.mtx, self.dist, None, new_mtx)
    
    def detect_shapes(self, frame):
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Rango para negro en HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        # Crear máscara para negro
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        squares = []
        lines = []
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            
            if len(corners) > 8:
                circles.append(contour)
            elif len(corners) == 4:
                squares.append(contour)
            else:
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                if abs(vx) < 0.1:  # Línea vertical
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

        # Mostrar patrones detectados
        self.draw_patterns(frame)
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
                cv2.circle(frame, center, self.size // 2, (0,0,0), thickness=5)
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
            frame = self.picam.capture_array()
            # frame = self.undistort_frame(frame)
            frame = self.check_pattern(frame)
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