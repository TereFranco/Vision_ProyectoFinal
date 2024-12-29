import cv2
import numpy as np
from picamera2 import Picamera2

def read_from_picam():
    """
    Configures the PiCam for real-time streaming.

    Returns:
        tuple: A tuple containing:
            - picam (Picamera2): The configured PiCam instance.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
    """
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    frame_width, frame_height = 1280, 720
    return picam, frame_width, frame_height

def initialize_kalman():
    """Initialize the Kalman filter."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32)
    return kf

def detect_traffic_light_color(frame):
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
    red_lower2 = np.array([160, 100, 100])
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

def pred_bbs(image: np.ndarray,net, IMAGE_WIDTH:int, IMAGE_HEIGHT:int, threshold_probability: float=0.9, iou_threshold: float=0.5, labels: list=[]):
    """
    Perform object detection on an input image using a pre-trained neural network, and return bounding boxes for detected objects.

    Args:
    - image: A numpy array representing the input image.
    - net: The pre-trained YOLO network.
    - threshold_probability: A float representing the minimum probability threshold for a detected object to be considered valid. Defaults to 0.9.
    - iou_threshold: A float representing the intersection over union threshold for Non-Maximum Suppression (NMS). Defaults to 0.5.
    - labels: A list of strings representing the class labels for the detected objects.

    Returns:
    - list: Detected bounding boxes, class IDs, and confidences.
    """
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)

    bboxes, confidences, class_ids = filter_output_probs(threshold_probability, outputs,IMAGE_WIDTH,IMAGE_HEIGHT)
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, threshold_probability, iou_threshold)
    return [(bboxes[i], class_ids[i], confidences[i]) for i in indices]


def filter_output_probs(threshold_probability: float, layer_outputs: tuple, IMAGE_WIDTH:int, IMAGE_HEIGHT:int) -> list:
    
    """
    Filters the detections from the output of a YOLO object detection model, by keeping only those with confidence scores
    above a specified threshold probability. Returns the bounding boxes, confidence scores, and class IDs of the filtered
    detections.

    Args:
        threshold_probability (float): The minimum confidence score for a detection to be included. Must be in the range [0, 1].
        layerOutputs (List): The output layers of a YOLO object detection model.

    Returns:
        Tuple[List[int], List[float], List[int]]: A tuple containing:
        - a list of bounding boxes, where each box is represented by a list of four integers (x_min, y_min, box_width, box_height);
        - a list of confidence scores, where each score is a float between 0 and 1;
        - a list of class IDs, where each ID is an integer corresponding to the index of the detected class label in the model's class labels list.
    """
    
    # Check validity of threshold_probability value
    if threshold_probability is None:
        raise ValueError("Invalid value for threshold_probability. Value cannot be `None`.")
    elif threshold_probability < 0 or threshold_probability > 1.0:
        raise ValueError(f"Cannot assign value {threshold_probability} to threshold_probability, value must be in range [0-1]")
    
    # Initialize variables to store object detection results
    boxes = []        # Bounding box coordinates for each detected object
    confidences = []  # Confidence score for each detected object
    class_ids = []    # Class ID for each detected object

    # Loop over the output layers from the YOLO model
    for output in layer_outputs:

        # Loop over each detection in the output
        for detection in output:

            # Extract the class probabilities and class ID for the current detection
            probabilities = detection[5:]
            class_id = np.argmax(probabilities)

            # Extract the confidence (probability) for the current detection
            confidence = probabilities[class_id]

            # Filter out weak detections with confidence below the given threshold_probability
            if confidence > threshold_probability:

                # Extract the bounding box coordinates and scale them to the size of the input image
                box = detection[0:4] * np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])

                # Convert the YOLO-format bounding box (center_x, center_y, width, height) to
                # the OpenCV-format bounding box (x_min, y_min, width, height) for drawing later
                center_x, center_y, box_width, box_height = box.astype('int')
                x_min = int(center_x - (box_width / 2))
                y_min = int(center_y - (box_height / 2))

                # Add the results to the output lists
                boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # Return the filtered results
    return boxes, confidences, class_ids


def correct_and_predict_picam(labels_path, cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    with open(labels_path, "r") as f:
        labels = f.read().strip().split("\n")

    picam, frame_width, frame_height = read_from_picam()
    kalman = initialize_kalman()

    while True:
        frame = picam.capture_array()

        detections = pred_bbs(frame, net, frame_width, frame_height,labels=labels)

        for bbox, class_id, confidence in detections:
            x, y, w, h = bbox

            # Update Kalman filter
            c_x, c_y = x + w / 2, y + h / 2
            prediction = kalman.predict()
            measurement = np.array([[c_x], [c_y]], dtype=np.float32)
            kalman.correct(measurement)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (int(prediction[0][0]), int(prediction[1][0])), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{labels[class_id]}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Detect traffic light
        traffic_light_color = detect_traffic_light_color(frame)
        cv2.putText(frame, f"Traffic Light: {traffic_light_color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam.stop()
    picam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    labels_path = "coco.names"
    cfg_path = "yolov4.cfg"
    weights_path = "yolov4.weights"
    correct_and_predict_picam(labels_path, cfg_path, weights_path)


