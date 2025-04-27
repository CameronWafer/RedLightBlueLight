import cv2
import numpy as np

class UI:
    # display start menu
    def __init__(self, window_name="Red Light Green Light Game"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show_start_screen(self, bg_path="C:\\Users\\HELIOS-300\\Downloads\\BackgroundRLBL.png"):
        # show start menu with custom background until enter pressed
        # load background image if available
        img = cv2.imread(bg_path)
        if img is None:
            background = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            background = cv2.resize(img, (640, 480))
        while True:
            frame = background.copy()
            # title text
            cv2.putText(
                frame,
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
                frame,
                "press ENTER to start",
                (170, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )
            # instructions text
            cv2.putText(
                frame,
                "q = quit | r = restart",
                (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # author credit
            author = "Author: PandaWafer"
            (tw, _), _ = cv2.getTextSize(author, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                frame,
                author,
                (640 - tw - 10, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(100) & 0xFF
            if key == 13:  # ENTER key
                break
        # reset window for game
        cv2.destroyWindow(self.window_name)
        cv2.namedWindow(self.window_name)
