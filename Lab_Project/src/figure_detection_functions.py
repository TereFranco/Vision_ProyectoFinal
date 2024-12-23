import os
import cv2
import numpy as np

# Ajustar los rangos de color negro para mejor detección
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 50])

PATTERN_PATH = "./Patterns/"
PATTERNS = {}
PATTERNS_GRAY = {}

def load_patterns():
    pattern_files = {
        'circle': 'circle_pattern.jpg',
        'square': 'square_pattern.jpg',
        'circle with line': 'circle_line_pattern.jpg',
        'square with line': 'square_line_pattern.jpg',
        'line': 'line_pattern.jpg'
    }
    
    for pattern_name, filename in pattern_files.items():
        path = os.path.join(PATTERN_PATH, filename)
        if os.path.exists(path):
            pattern = cv2.imread(path)
            if pattern is not None:
                PATTERNS[pattern_name] = pattern
                PATTERNS_GRAY[pattern_name] = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

load_patterns()

# Patrones de secuencia
FIRST_PATTERN_SEQUENCE = ["line", "circle", "square"]
SECOND_PATTERN_SEQUENCE = ["circle with line", "square"]
THIRD_PATTERN_SEQUENCE = ["circle", "square with line"]
FOURTH_PATTERN_SEQUENCE = ["circle", "square", "line"]

PATTERNS_COLORS = {
    'circle': (255, 0, 0),
    'square': (0, 255, 0),
    'circle with line': (255, 0, 255),
    'square with line': (0, 255, 255),
    'line': (0, 0, 255)
}

def preprocess_frame(frame):
    frame_blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv_frame = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv_frame, LOWER_BLACK, UPPER_BLACK)
    
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.erode(black_mask, kernel, iterations=1)
    black_mask = cv2.dilate(black_mask, kernel, iterations=1)
    
    return black_mask

def figure_detection(frame, pattern_sequence, last_pattern_detected, min_confidence=0.75):
    black_mask = preprocess_frame(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_pattern = None
    max_confidence = 0
    detected_location = None
    
    for pattern_name, pattern_gray in PATTERNS_GRAY.items():
        result = cv2.matchTemplate(frame_gray, pattern_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > max_confidence and max_val >= min_confidence:
            max_confidence = max_val
            detected_pattern = pattern_name
            detected_location = max_loc
    
    if detected_pattern:
        h, w = PATTERNS_GRAY[detected_pattern].shape
        cv2.rectangle(frame, 
                     detected_location, 
                     (detected_location[0] + w, detected_location[1] + h),
                     PATTERNS_COLORS[detected_pattern], 
                     2)
        
        cv2.putText(frame, 
                    f"{detected_pattern} ({max_confidence:.2f})", 
                    (detected_location[0], detected_location[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    PATTERNS_COLORS[detected_pattern],
                    2)
        
        if detected_pattern != last_pattern_detected:
            pattern_sequence.append(detected_pattern)
    
    edges = cv2.Canny(black_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if detected_pattern:
        cv2.drawContours(frame, filtered_contours, -1, PATTERNS_COLORS[detected_pattern], 1)
    
    return frame, detected_pattern