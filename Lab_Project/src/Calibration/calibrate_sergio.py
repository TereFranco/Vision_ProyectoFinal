import time
import numpy as np
import cv2
import glob
import imageio
from typing import List
# from picamera2 import Picamera2, Preview


CHESSBOARD_PATH = "Images_calibration/Input/"

def create_imgs_path(num_images: int):
    imgs_path = []
    for i in range(num_images):
        imgs_path.append(f"Images_calibration/Input/captured_image_{i}.jpg")
    return imgs_path

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

def take_chessboard_images(n_images):
    """
    Takes n_images of a chessboard every 2 seconds
    """
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)
    picam2.start_preview(Preview.QTGL)

    picam2.start()

    for i in range(n_images):
        time.sleep(2)
        picam2.capture_file(CHESSBOARD_PATH + "chessboard" + str(i) + ".jpg")

    picam2.stop()


def get_calibration_points(criteria, objp):
    """
    Get the calibration points from the images

    Parameters
    ----------
    criteria : termination criteria
    objp : object points

    Returns
    -------
    objpoints : object points
    imgpoints : image points
    shape : image size
    """
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    images = create_imgs_path(9)
    #images = load_images(imgs_path)
    #images = glob.glob(CHESSBOARD_PATH + "*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints, imgpoints, gray.shape[::-1]


def calibrate():
    """
    Calculates the camera matrix and distortion coefficients
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,
    # (9,10,0). Grid is 10x11
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints, imgpoints, image_size = get_calibration_points(criteria, objp)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Print the camera calibration parameters and the rmse
    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)
    print("RMSE:")
    print(ret)
    # Save the camera calibration parameters
    np.savez("calibration.npz", rmse=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


if __name__ == "__main__":
    # take_chessboard_images(25)
    calibrate()
