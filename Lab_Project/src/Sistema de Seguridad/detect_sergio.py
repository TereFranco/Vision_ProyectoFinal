import cv2
from picamera2 import Picamera2
import numpy as np


class Figure:
    def __init__(self, figure_type, color_name, color, n_vertices, color_threshold=40):
        self.figure_type = figure_type
        self.color_name = color_name
        self.color = np.array(color)
        self.n_vertices = n_vertices
        self.color_threshold = color_threshold

    def __eq__(self, other):
        if not isinstance(other, Figure):
            return False
        return (
            self.figure_type == other.figure_type and self.color_name == other.color_name
        )

    def __repr__(self):
        return f"Figure({self.figure_type}, {self.color_name}, {self.color}, {self.n_vertices})"

    def __str__(self):
        return f"{self.color} {self.figure_type}"

    def detect(self, img):
        """
        Returns True if the image contains the figure
        """

        # Remove alpha channel
        img_rgb = img[:, :, :3]

        # Blur without changing color
        ks = 5
        blur = cv2.GaussianBlur(img_rgb, (ks, ks), 0)

        # Use mask with euclidean distance
        mask = np.linalg.norm(blur - self.color, axis=-1) < self.color_threshold
        thresh = np.zeros_like(mask, dtype=np.uint8)
        thresh[mask] = 255

        cv2.imshow("thresh", thresh)

        # Erode to remove noise
        ks = 9
        iters = 6
        kernel = np.ones((ks, ks), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=iters)

        # Dilate to recover original size
        # dilated = cv2.dilate(eroded, kernel, iterations=iters)

        # Find the contours
        contours, _ = cv2.findContours(
            eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.imshow("dilated", eroded)

        # Match with all contours
        for contour in contours:
            corners = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True
            )

            # Match figure type
            if len(corners) == self.n_vertices:
                return True

        return False

    def plot_on_image(self, img, center, size):
        """
        Plots a figure on the image
        """

        # Get vertices
        vertices = []
        for i in range(self.n_vertices):
            angle = 2 * np.pi * i / self.n_vertices
            # Add up angle to even number of vertices
            angle += ((self.n_vertices + 1) % 2) * np.pi / self.n_vertices
            # First vertex is at the top
            angle -= np.pi / 2
            x = int(center[0] + size * np.cos(angle))
            y = int(center[1] + size * np.sin(angle))
            vertices.append((x, y))

        # Plot figure
        c = tuple(self.color.tolist())
        cv2.fillPoly(img, np.array([vertices]), c)


def get_picam2():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"},
    )
    picam2.configure(config)
    return picam2


if __name__ == "__main__":
    # Create figures
    valid_figures = [
        Figure("quadrilateral", "red", (50, 40, 140), 4),
        Figure("quadrilateral", "yellow", (35, 155, 172), 4),
        Figure("triangle", "blue", (140, 85, 45), 3),
        Figure("pentagon", "green", (50, 85, 55), 5),
        # Figure("triangle", "red", (0, 0, 178), 3),
        # Figure("triangle", "yellow", (0, 145, 200), 3),
        # Figure("quadrilateral", "green", (30, 125, 15), 4),
        # Figure("pentagon", "blue", (249, 54, 0), 5),
    ]

    # Try to detect a quadrilateral
    picam2 = get_picam2()
    picam2.start()

    while cv2.waitKey(1) != ord("q"):
        img = picam2.capture_array()

        # Print colo in the center
        center = (320, 240)
        print(img[center[1], center[0]])

        detected = valid_figures[3].detect(img)

        if detected:
            cv2.rectangle(img, (0, 0), (640, 480), (0, 255, 0), 3)

        cv2.imshow("frame", img)

    picam2.stop()
    cv2.destroyAllWindows()
