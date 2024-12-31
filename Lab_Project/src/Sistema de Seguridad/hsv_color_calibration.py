import cv2
import numpy as np
from picamera2 import Picamera2

def nothing(x):
    pass

# Inicializar cámara
picam = Picamera2()
picam.preview_configuration.main.size = (640, 480)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()

cv2.namedWindow('Tracking')
cv2.createTrackbar('LH', 'Tracking', 0, 179, nothing)
cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('UH', 'Tracking', 179, 179, nothing)
cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

while True:
    frame = picam.capture_array()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('LH', 'Tracking')
    l_s = cv2.getTrackbarPos('LS', 'Tracking')
    l_v = cv2.getTrackbarPos('LV', 'Tracking')
    u_h = cv2.getTrackbarPos('UH', 'Tracking')
    u_s = cv2.getTrackbarPos('US', 'Tracking')
    u_v = cv2.getTrackbarPos('UV', 'Tracking')

    l_g = np.array([l_h, l_s, l_v])
    u_g = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_g, u_g)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print(f"Lower HSV: [{l_h}, {l_s}, {l_v}]")
        print(f"Upper HSV: [{u_h}, {u_s}, {u_v}]")
        break

cv2.destroyAllWindows()