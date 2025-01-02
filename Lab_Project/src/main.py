from Sistema_de_Seguridad.Authenticator import Authenticator
from Sistema_de_Seguridad.Figure import Figure
from Sistema_de_Seguridad.initialize_picam import initialize
from Sistema_Propuesto.tracker import Tracker
import numpy as np

if __name__ == "__main__":
    #our password
    password = [
        Figure("circle", "blue", (0,112, 192), np.array([100,130,0]), np.array([179,255,255]), 9), 
        Figure("square", "green", (0,176,80), np.array([40, 100, 50]), np.array([100, 255, 255]), 4),
        Figure("triangle", "red", (255,0,0), np.array([0,155,113]), np.array([7,210,155]), 3),
        Figure("pentagon", "purple", (112,48,160), np.array([123,125,23]), np.array([161,255,178]), 5)
    ]

    picam = initialize()

    authenticator = Authenticator(picam,password)
    authenticator.check_password()

    tracker = Tracker(picam)

    print("Presiona 's' para seleccionar el ROI y empezar el seguimiento, y 'q' para detener.")

    tracker.run()

    picam.stop()

    print("Seguimiento finalizado.")