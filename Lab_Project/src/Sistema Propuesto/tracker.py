import cv2
import numpy as np
from picamera2 import Picamera2

def read_from_picam(): #La picam
    """
    Configures the PiCam for real-time streaming.

    Returns:
        generator: A generator that yields frames from the PiCam.
    """
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    return picam

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

def initialize_kalman(): #inicializar kalman de practica 4
    """Initialize the Kalman filter."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32)
    return kf

def correct_and_predict(frame, kf, crop_hist, track_window): #predecir kalman de la practica 4 tambien, 
    """
    Corrects the Kalman filter with the current measurement and predicts the next state.

    Args:
        frame (numpy.ndarray): The current frame read from the video.
        kf (cv2.KalmanFilter): The initialized Kalman filter.
        crop_hist (numpy.ndarray): The histogram of the cropped region of interest (ROI).
        track_window (tuple): The initial tracking window (x, y, width, height).

    Returns:
        tuple: (bool, tuple) Success flag and updated tracking window.
    """
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

    input_frame = frame.copy()
    img_hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)

    img_bproject = cv2.calcBackProject([img_hsv], [0], crop_hist, [0, 180], 1)

    ret, track_window = cv2.meanShift(img_bproject, track_window, term_crit)
    x, y, w, h = track_window

    c_x = x + w / 2
    c_y = y + h / 2

    prediction = kf.predict()

    measurement = np.array([[c_x], [c_y]], np.float32)
    kf.correct(measurement)

    cv2.rectangle(input_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for bounding box
    cv2.circle(input_frame, (int(prediction[0]), int(prediction[1])), 5, (0, 0, 255), -1)  # Red for prediction
    cv2.circle(input_frame, (int(c_x), int(c_y)), 5, (0, 255, 0), -1)  # Green for measurement

    # Detectar el color del semáforo
    traffic_light_color = detect_traffic_light_color(frame)
    cv2.putText(input_frame, f"Traffic Light: {traffic_light_color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    print(f"Traffic Light Color: {traffic_light_color}")

    cv2.imshow('Tracker', input_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return False, track_window

    return True, track_window

if __name__ == "__main__":
    picam = read_from_picam()

    print("Presiona 's' para seleccionar el ROI y empezar el seguimiento, y 'q' para detener.")

    is_tracking = False

    try:
        while True:
            frame = picam.capture_array()
            cv2.imshow("PiCam", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not is_tracking:
                #manualmente decirle que tiene que seguir al objeto, el coche en nustro caso
                print("Selecciona el ROI para comenzar el seguimiento.")
                x, y, w, h = cv2.selectROI("Selecciona el objeto", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyAllWindows()

                track_window = (x, y, w, h)
                crop = frame[y:y + h, x:x + w].copy()
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                crop_hist = cv2.calcHist([hsv_crop], [0], None, [180], [0, 180])
                cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

                kf = initialize_kalman()
                is_tracking = True

            if is_tracking:
                success, track_window = correct_and_predict(frame, kf, crop_hist, track_window)
                if not success:
                    break

            if key == ord('q'):
                print("Tecla 'q' presionada. Deteniendo...")
                break

    finally:
        picam.stop()
        picam.close()
        cv2.destroyAllWindows()
        print("Seguimiento finalizado.")