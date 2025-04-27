import cv2
import numpy as np

class UI:
    # display start menu
    def __init__(self, window_name="Red Light Green Light Game"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show_start_screen(self):
        # keeps showing a start screen until the user presses ENTER
        while True:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            # title text
            cv2.putText(
                blank,
                "Red Light, Green Light",
                (80, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )
            # prompt text
            cv2.putText(
                blank,
                "press enter to start",
                (170, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )
            cv2.imshow(self.window_name, blank)
            key = cv2.waitKey(100) & 0xFF
            if key == 13:  # ENTER key
                break
        # reset window for the game
        cv2.destroyWindow(self.window_name)
        cv2.namedWindow(self.window_name)
