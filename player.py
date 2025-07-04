import cv2
import numpy as np

class Player:
    def __init__(self, id: int, face_filter: np.ndarray | None = None):
        self.id = id
        self.face_filter = face_filter
        self.moving_frame_count = 0
        self.face_size = 0 # for win condition

    def is_moving(self, current_face: np.ndarray) -> bool:
        if self.face_filter is None or current_face is None:
            return False

        # Resize current face to match stored size just in case
        current_face = cv2.resize(current_face, (100, 100))

        # Calculate absolute difference
        diff = cv2.absdiff(self.face_filter, current_face)

        # Compute mean pixel difference
        score = np.mean(diff)

        print(f"[Player {self.id}] Difference Score: {score}")

       # Cooldown logic: only eliminate if the difference score is high for several frames
        if score > 90:  # can tweak this threshold
            self.moving_frame_count += 1
        else:
            self.moving_frame_count = 0  # reset if they stopped moving

        # Only flag if moved in 3+ frames in a row
        return self.moving_frame_count >= 3

    def update_face_size(self, size: int):
        self.face_size = size
        print(f"[Player {self.id}] Updated face_size: {self.face_size}")

    def is_won(self, winning_threshold: int = 200) -> bool:
        print(f"[Player {self.id}] Checking win: face_size={self.face_size}, threshold={winning_threshold}")
        return self.face_size >= winning_threshold
