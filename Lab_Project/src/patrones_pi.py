import cv2
import numpy as np

# Cargar y procesar imágenes de referencia
def load_reference_shapes(ref_images):
    references = {}
    for name, path in ref_images.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        references[name] = contours[0]
    return references

# Comparar contornos
def match_shapes(contour, reference_shapes):
    for name, ref_contour in reference_shapes.items():
        similarity = cv2.matchShapes(contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < 0.1:
            return name
    return None

# Detectar formas en un frame
def detect_shapes_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    _, binary = cv2.threshold(sobel_normalized, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detectar posición del lápiz
def detect_pencil_position(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])  # Ajusta el rango de color del lápiz aquí
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pencil_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(pencil_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

# Obtener centro de la forma
def get_shape_center(shape_name, shapes, reference_shapes):
    for contour in shapes:
        matched_shape = match_shapes(contour, reference_shapes)
        if matched_shape == shape_name:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

# Procesar video desde la webcam
def process_webcam_with_phases(ref_images, output_path=None):
    # Cargar referencias (círculo y cuadrado)
    reference_shapes = load_reference_shapes(ref_images)

    # Estado inicial
    current_phase = 0  # Comenzamos fuera de las fases (0 = inicio)

    # Iniciar captura desde la webcam
    cap = cv2.VideoCapture(0)  # Cambiar a '1' si usas una segunda cámara
    if not cap.isOpened():
        print("No se pudo acceder a la webcam.")
        return

    # Configuración para guardar el video (opcional)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar formas y posición del lápiz
        shapes = detect_shapes_in_frame(frame)
        detected_shapes = [match_shapes(c, reference_shapes) for c in shapes]
        pencil_pos = detect_pencil_position(frame)

        if pencil_pos and "circle" in detected_shapes and "square" in detected_shapes:
            circle_pos = get_shape_center("circle", shapes, reference_shapes)
            square_pos = get_shape_center("square", shapes, reference_shapes)

            dx, dy = pencil_pos[0] - circle_pos[0], pencil_pos[1] - circle_pos[1]
            circle_radius = int(np.sqrt(cv2.contourArea(shapes[0]) / np.pi))

            x, y, w, h = cv2.boundingRect(shapes[1])
            square_left, square_right = x, x + w
            square_top, square_bottom = y, y + h

            # Determinar la fase actual
            if pencil_pos[0] < min(circle_pos[0], square_pos[0]):
                phase = 1
            elif dx**2 + dy**2 <= circle_radius**2:
                phase = 2
            elif square_left <= pencil_pos[0] <= square_right and square_top <= pencil_pos[1] <= square_bottom:
                phase = 3
            elif pencil_pos[0] > max(circle_pos[0], square_pos[0]):
                phase = 4
            else:
                phase = current_phase  # Mantén la fase si no cambió

            # Validar si la transición es válida
            if phase > current_phase:
                print(f"Transición: Fase {current_phase} → Fase {phase}")
                current_phase = phase
            elif phase < current_phase:
                print(f"¡Fallo! Transición inválida: Fase {current_phase} → Fase {phase}")
                current_phase = -1
                break
            elif phase == current_phase:
                print(f"Fase actual: {current_phase}, sin cambio.")

            # Mostrar la fase actual en el frame
            if current_phase > 0:
                cv2.putText(frame, f"Fase: {current_phase}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Fallo detectado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            print("Formas no detectadas o lápiz no encontrado.")

        # Guardar el frame procesado si se solicita
        if output_path:
            out.write(frame)

        # Mostrar el frame
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    # Resultado final
    if current_phase == 4:
        print("Contraseña correcta.")
    else:
        print("Contraseña incorrecta. No se ha pasado efectivamente por los patrones.")

if __name__ == "__main__":
    ref_images = {
    "circle": "Images/pattern_image_4.jpg",  # Imagen del círculo de referencia
    "square": "Images/pattern_image_10.jpg"  # Imagen del cuadrado de referencia
    }   
    # Procesar en tiempo real y guardar el video procesado (opcional)
    process_webcam_with_phases(ref_images, output_path="output_video.avi")