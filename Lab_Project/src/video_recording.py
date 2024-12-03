from picamera2 import Picamera2, Preview
import cv2

def record_video(picam: Picamera2, output_file: str):
    video_config = picam.create_video_configuration(main={"size": (1280, 720), "format": "MJPEG"})
    picam.configure(video_config)

    picam.start()

    is_recording = False

    try:
        print("Presiona 's' para empezar a grabar, y 'q' para detener.")

        while True:
            frame = picam.capture_array()
            cv2.imshow("picam", frame)

            # Capturar la tecla presionada
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and not is_recording:
                print("Grabando video. Presiona 'q' para detener.")
                picam.start_recording(output_file)

            if key == ord('q') and is_recording:
                print("Tecla 'q' presionada. Deteniendo la grabación...")

    finally:
        # Detener la grabación y limpiar
        picam.stop_recording()
        picam.stop()
        picam.close()
        cv2.destroyAllWindows()
        print(f"Video guardado como {output_file}")



def main():
    picam = Picamera2()
    output_file = "patron_correcto.mjpeg"

    record_video(picam, output_file)


if __name__ == "__main__":
    main()