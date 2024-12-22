import os
import cv2
from skimage import feature
import numpy as np

LOWER_BLACK = (0, 90, 0)
UPPER_BLACK = (255, 255, 255)

PATTERN_PATH = "./Patterns/"
CIRCLE_PATTERN = cv2.imread(os.path.join(PATTERN_PATH,"circle_pattern.jpg"))
SQUARE_PATTERN = cv2.imread(os.path.join(PATTERN_PATH,"square_pattern.jpg"))
CIRCLE_LINE_PATTERN = cv2.imread(os.path.join(PATTERN_PATH,"circle_line_pattern.jpg"))
SQUARE_LINE_PATTERN = cv2.imread(os.path.join(PATTERN_PATH,"square_line_pattern.jpg"))
LINE_PATTERN = cv2.imread(os.path.join(PATTERN_PATH,"line_pattern.jpg"))
PATTERNS = [CIRCLE_PATTERN,SQUARE_PATTERN,CIRCLE_LINE_PATTERN,SQUARE_LINE_PATTERN,LINE_PATTERN]
PATTERNS_NAMES = ["circle","square","circle with line","square with line","line"]

# Gray patterns
CIRCLE_PATTERN_GRAY = cv2.cvtColor(CIRCLE_PATTERN,cv2.COLOR_BGR2GRAY)
SQUARE_PATTERN_GRAY = cv2.cvtColor(SQUARE_PATTERN,cv2.COLOR_BGR2GRAY)
CIRCLE_LINE_PATTERN_GRAY = cv2.cvtColor(CIRCLE_LINE_PATTERN,cv2.COLOR_BGR2GRAY)
SQUARE_LINE_PATTERN_GRAY = cv2.cvtColor(SQUARE_LINE_PATTERN,cv2.COLOR_BGR2GRAY)
LINE_PATTERN_GRAY = cv2.cvtColor(LINE_PATTERN,cv2.COLOR_BGR2GRAY)
PATTERNS_GRAY = [CIRCLE_PATTERN_GRAY,SQUARE_PATTERN_GRAY,CIRCLE_LINE_PATTERN_GRAY,SQUARE_LINE_PATTERN_GRAY,LINE_PATTERN_GRAY]

# Patterns sequences
FIRST_PATTERN_SEQUENCE = ["line","circle","square"]
SECOND_PATTERN_SEQUENCE = ["circle with line","square"]
THIRD_PATTERN_SEQUENCE = ["circle","square with line"]
FOURTH_PATTERN_SEQUENCE = ["circle","square","line"]

# Threshold
THRESHOLD_MATCH_TEMPLATE = 0.8

# Colors
PATTERNS_COLORS = [[255,0,0],[0,255,0],[255,0,255],[0,255,255],[0,0,255]]

# Fonts
FONT = cv2.FONT_HERSHEY_SIMPLEX

def figure_detection(frame, pattern_sequence, last_pattern_detected):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv_frame, LOWER_BLACK, UPPER_BLACK)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_pattern = False

    for i, pattern in enumerate(PATTERNS_GRAY):
        result = cv2.matchTemplate(frame_gray, pattern, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= THRESHOLD_MATCH_TEMPLATE:
            detected_pattern = True
            top_left = max_loc
            cv2.putText(frame, PATTERNS_NAMES[i], (top_left[0], top_left[1] - 10), FONT, 0.5, (255,255,255), 2, cv2.LINE_AA)
            if PATTERNS_NAMES[i] != last_pattern_detected:
                pattern_sequence.append(PATTERNS_NAMES[i])
            break

    img_mask = feature.canny(black_mask)
    img_mask = img_mask.astype(np.uint8) * 255
    borders = np.where(img_mask == 255)
    if detected_pattern:
        # Draw borders on the original frame
        for y, x in zip(borders[0], borders[1]):
            frame[y, x] = PATTERNS_COLORS[i] 
    
    return frame, pattern