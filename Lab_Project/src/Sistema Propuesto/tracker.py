import cv2
import numpy as np
from picamera2 import Picamera2

class Tracker:
    def __init__(self):
        self.picam = Picamera2()
        self.picam.preview_configuration.main.size = (640, 480)
        self.picam.preview_configuration.main.format = "RGB888"
        self.picam.preview_configuration.align()
        self.picam.configure("preview")
        self.picam.start()
        self.is_tracking = False
        self.kf = self.create_kalman_filter()
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
        self.meanshift_threshold = 15

    def create_kalman_filter(self): #inicializar kalman de practica 4
        """Initialize the Kalman filter."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)

        return kf
    
    def initialize_kalman(self, frame):
        self.select_roi(frame)
        x, y, w, h = self.bbox

        # Compute the center of the object
        cx = x + w/2
        cy = y + h/2

        # Initialize the state of the Kalman filter
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        
        # Ajustar los parámetros de ruido para un seguimiento más estable
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05  # Reduced for more stable movement
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  # Reduced to trust measurements more

        # Initialize the covariance matrix with moderate uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

        # Crop the object
        crop = frame[y:y+h, x:x+w].copy()
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        self.crop_hist = cv2.calcHist([hsv_crop], [0], mask=None, histSize=[180], ranges=[0, 180])
        cv2.normalize(self.crop_hist, self.crop_hist, 0, 255, cv2.NORM_MINMAX)

    def select_roi(self, frame):
        print("Selecciona el ROI para comenzar el seguimiento.")
        self.bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")

    def correct_and_predict(self, frame): #predecir kalman de la practica 4 tambien, 
        """
        Corrects the Kalman filter with the current measurement and predicts the next state.

        Args:
            frame (numpy.ndarray): The current frame read from the video.

        Returns:
            numpy.ndarray: The frame with tracking visualization.
        """
        input_frame = frame.copy()
        img_hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
        
        # Get current window
        x, y, w, h = self.bbox
        
        # Calculate back projection
        img_bproject = cv2.calcBackProject([img_hsv], [0], self.crop_hist, [0, 180], 1)
        
        # Mejorar la detección aplicando un umbral y limpieza
        _, img_bproject = cv2.threshold(img_bproject, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_bproject = cv2.morphologyEx(img_bproject, cv2.MORPH_OPEN, kernel)
        
        # Apply meanShift
        ret, track_window = cv2.meanShift(img_bproject, self.bbox, self.term_crit)
        
        # Calcular confianza basada en la proyección posterior y el área
        x_ms, y_ms, w_ms, h_ms = track_window
        roi = img_bproject[y_ms:y_ms+h_ms, x_ms:x_ms+w_ms]
        if roi.size == 0:  # Si el ROI está fuera de la imagen
            return input_frame
        
        # Calcular el porcentaje de píxeles blancos en el ROI
        white_pixels = np.sum(roi > 0)
        total_pixels = roi.size
        confidence = (white_pixels / total_pixels) * 100
        
        # Si la confianza es muy baja, no mostrar nada
        if confidence < 15:  # Ajusta este umbral según necesites
            return input_frame
            
        # Get Kalman prediction
        prediction = self.kf.predict()
        
        # Calcular el centro actual
        c_x = x_ms + w_ms/2
        c_y = y_ms + h_ms/2
        
        # Verificar que la predicción no se aleje demasiado del último punto conocido
        dist_to_last = np.sqrt((prediction[0][0] - c_x)**2 + (prediction[1][0] - c_y)**2)
        if dist_to_last > w_ms:  # Si la predicción se aleja más que el ancho del objeto
            prediction[0][0] = c_x
            prediction[1][0] = c_y
        
        # Update measurement and correct Kalman filter
        measurement = np.array([[c_x], [c_y]], np.float32)
        self.kf.correct(measurement)
        
        # Dibujar el rectángulo de tracking
        img2 = cv2.rectangle(input_frame, (x_ms, y_ms), (x_ms+w_ms, y_ms+h_ms), (255,0,0), 2)
        
        # Dibujar los puntos de predicción y medición con los colores originales
        cv2.circle(img2, (int(prediction[0][0]), int(prediction[1][0])), 5, (0, 0, 255), -1)  # Rojo para predicción
        cv2.circle(img2, (int(c_x), int(c_y)), 5, (0, 255, 0), -1)  # Verde para posición actual
        
        # Actualizar bbox para el siguiente frame
        self.bbox = track_window
        
        return img2
    
    def run(self):
        while True:
            frame = self.picam.capture_array()
            

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not self.is_tracking:
                self.initialize_kalman(frame)
            
                self.is_tracking = True

            if self.is_tracking:
                frame = self.correct_and_predict(frame)

            cv2.imshow("picam", frame)

            if key == ord('q'):
                print("Tecla 'q' presionada. Deteniendo...")
                break

        self.picam.stop()
        cv2.destroyAllWindows()

def detect_traffic_light_color(frame): #de chat para la parte opcional, para detectar los colores de señales de trafico, rojo y verde
    """
    Detect the color of traffic lights in the given frame.

    Args:
        frame (numpy.ndarray): The current frame of the video.

    Returns:
        str: The detected color ("Red", "Green", or "Unknown").
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])# dos de rojo q lo dice chat
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels and red_pixels > 50:
        return "Red"
    elif green_pixels > red_pixels and green_pixels > 50:
        return "Green"
    else:
        return "Unknown"





if __name__ == "__main__":
    tracker = Tracker()

    print("Presiona 's' para seleccionar el ROI y empezar el seguimiento, y 'q' para detener.")

    try:
        tracker.run()

    finally:
        print("Seguimiento finalizado.")