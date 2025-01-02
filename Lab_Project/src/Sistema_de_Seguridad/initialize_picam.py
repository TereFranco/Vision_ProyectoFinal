from picamera2 import Picamera2

def initialize() -> Picamera2:
    """
    Initializes the Picamera2 with specific configurations.

    Returns:
        Picamera2: The initialized Picamera2 object.
    """
    picam = Picamera2()
    picam.preview_configuration.main.size = (640, 480)
    picam.preview_configuration.main.format = "RGB888"  #entiendo que esto está bien? HARIA FALTA CAMBIAR LOS FAMRES DE COLOR?
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    return picam