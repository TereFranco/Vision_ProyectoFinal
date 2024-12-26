import cv2
from picamera2 import Picamera2
from detect_pattern.detect import Figure


def get_picam2():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"},
    )
    picam2.configure(config)
    return picam2


def enter_password(picam2, password: list, valid_figures: list):
    picam2.start()

    password_is_correct = None

    # Stable flanks
    prev_state = None  # Detected figure
    state = None  # Detected figure
    n_lags = 10
    lagged_states = []  # Detected figures

    sequence = []
    next_character = 0

    while cv2.waitKey(1) != ord("q") and password_is_correct is None:
        # Capture image
        im = picam2.capture_array()

        # Convert to RGB
        frame = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect figures
        figure_found = None
        for figure in valid_figures:
            if figure.detect(frame):
                figure_found = figure
                # print("Detected!", lagged_states)
                break

        lagged_states.append(figure_found)
        if len(lagged_states) > n_lags:
            lagged_states.pop(0)

        # Update state
        if len(lagged_states) == n_lags:
            f0 = lagged_states[0]
            # If all figures are the same
            if all(f0 == f for f in lagged_states):
                state = f0

        # Detect edges
        if prev_state != state:
            if state is not None:
                print(f"{state} detected")
                sequence.append(state)
                next_character += 1
            else:
                print("Figure disappeared")

        # Update previous state
        prev_state = state

        # Check password
        if len(sequence) == len(password):
            if sequence == password:
                print("Correct password")
                password_is_correct = True
            else:
                print("Incorrect password")
                password_is_correct = False

        # Plot sequence
        for i, figure in enumerate(sequence):
            figure.plot_on_image(frame, (50 + 70 * i, 50), 30)

        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # If password is correct, show camera with green border for 5 seconds
    seconds = 2
    color = (0, 255, 0) if password_is_correct else (0, 0, 255)
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < seconds:
        im = picam2.capture_array()
        # remove alpha channel
        frame = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0, 0), (640, 480), color, 10)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

    return password_is_correct


def authenticate(picam, password, valid_figures):
    while not enter_password(picam, password, valid_figures):
        print("Incorrect password. Try again.")


if __name__ == "__main__":
    picam2 = get_picam2()
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

    password = [
        valid_figures[0],
        valid_figures[1],
        valid_figures[2],
        valid_figures[3],
    ]
    authenticate(picam2, password, valid_figures)
