from picamera2 import Picamera2, Preview
import cv2

def record_video(picam: Picamera2, output_file: str):
    video_config = picam.create_video_configuration(main={"format": "MJPEG"})
    picam.configure(video_config)

    picam.start()

    try:
        print("Grabando video. Presiona 'q' para detener.")
        picam.start_recording(output_file)

        while True:
            # Capturar la tecla presionada
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
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