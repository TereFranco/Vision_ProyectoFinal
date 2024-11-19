import cv2
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    i = 0
    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('f'):
            cv2.imwrite(f"captured_image{i}.jpg", frame)
            print("Imagen capturada y guardada como 'captured_image.jpg'")
            i += 1
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()