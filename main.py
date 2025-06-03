from constants import *
from player import Player
from datetime import datetime, timedelta
from random import random
import cv2
import numpy as np
import time
import math
import tensorflow as tf
import tensorflow_hub as hub

# Import updated cv_interface functions directly
from cv_interface import get_player_filters, check_player_movement, check_player_winning, preprocess_frame
from init_player_faces import extract_faces_from_keypoints

# MoveNet MultiPose model
movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet_input_size = 256

# Start Screen
class UI:
    def __init__(self, window_name="Red Light Green Light Game"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)

    def show_start_screen(self, bg_path="C:\\Users\\HELIOS-300\\Downloads\\BackgroundRLBL.png"):
        # use image if path exist, otherwise black background
        img = cv2.imread(bg_path)
        if img is None:
            background = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            background = cv2.resize(img, (640, 480))

        while True:
            frame = background.copy()
            cv2.putText(frame, "Red Light, Green Light",
                        (80, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "press ENTER to start",
                        (170, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "q = quit",
                        (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            author = "Author: Cal Poly SLO"
            (tw, _), _ = cv2.getTextSize(author,
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, 2)
            cv2.putText(frame, author,
                        (640 - tw - 10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(100) & 0xFF
            if key == 13:  # "enter" key
                break

        cv2.destroyWindow(self.window_name)

class Game:
    def __init__(self):
        self.state: GameState = GameState.READ_FACES
        self.players_playing: dict[int, Player] = {}
        self.players_won: dict[int, Player] = {}
        self.players_lost: dict[int, Player] = {}
        self.time_for_next_state = datetime.now()
        self.cap = None  # Camera will be initialized in run()
        self.start_time = time.time() # for pulsing red light phase

    def run(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("❌ Could not open video capture")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("❌ Failed to grab frame")
                break

            match self.state:
                case GameState.READ_FACES:
                    self.read_faces()
                case GameState.GREEN_LIGHT:
                    self.green_light()
                case GameState.RED_LIGHT:
                    self.red_light()
                case GameState.END_GAME:
                    self.end_game()
                    break

            border_color = STATE_COLORS[self.state]
            height, width = frame.shape[:2]

            # draw border and text
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color.value, thickness=10)
            cv2.putText(frame, f"State: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            if self.state in [GameState.RED_LIGHT, GameState.GREEN_LIGHT]:
                time_remaining_str = str(self.time_for_next_state - datetime.now())
                cv2.putText(frame, f"{time_remaining_str[time_remaining_str.index(':') + 1:]}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            cv2.putText(frame, f"Players: {len(self.players_playing)}", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, border_color.value, 2)

            # show current game state as a traffic light
            light_w, light_h = 60, 140
            margin = 20
            x0 = width - light_w - margin
            y0 = margin

            # background box
            cv2.rectangle(frame, (x0, y0), (x0 + light_w, y0 + light_h), (50, 50, 50), -1)

            # circle parameters
            radius = (light_w // 2) - 10
            cx = x0 + light_w // 2
            red_y = y0 + radius + 10
            green_y = y0 + light_h - radius - 10
            dark = (30, 30, 30)

            # light on/off
            if self.state == GameState.GREEN_LIGHT:
                cv2.circle(frame, (cx, red_y), radius, dark, -1)  # red off
                cv2.circle(frame, (cx, green_y), radius, (0, 255, 0), -1)  # green on
            else:
                cv2.circle(frame, (cx, red_y), radius, (0, 0, 255), -1)  # red on
                cv2.circle(frame, (cx, green_y), radius, dark, -1)  # green off

            # pulsing red light phase
            if self.state == GameState.RED_LIGHT:
                alpha = 0.2 + 0.1 * math.sin(2 * math.pi * (time.time() - self.start_time))
                overlay = np.full(frame.shape, (0, 0, 255), dtype=np.uint8)  # BGR: red
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

            cv2.imshow('Live Feed', frame)

            # Temp key listeners for quitting or ending
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                print("🔚 Forcing end game...")
                self.state = GameState.END_GAME

        self.cap.release()
        cv2.destroyAllWindows()

    def read_faces(self) -> None:
        print("🔍 Reading player filters...")
        self.players_playing = get_player_filters(self.cap)
        print(f"✅ Loaded {len(self.players_playing)} player(s)")
        self.to_green_light_state()

    def update_faces(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        input_img = preprocess_frame(frame)
        outputs = movenet.signatures["serving_default"](tf.constant(input_img))
        people = outputs["output_0"].numpy()
        faces_with_sizes = extract_faces_from_keypoints(frame, people)

        if not faces_with_sizes:
            return  # No faces detected; skip update

        # explicitly handle unpacking
        for face_data, player in zip(faces_with_sizes, self.players_playing.values()):
            if len(face_data) == 2:
                face, size = face_data
                player.update_face_size(size)
            else:
                print("Warning: Unexpected face_data structure:", face_data)

    def green_light(self):
        self.update_faces()

        if self.time_for_next_state < datetime.now() and self.players_playing:
            self.to_red_light_state()
        else:
            check_player_winning(self.players_playing, self.players_won)

        if not self.players_playing:
            self.state = GameState.END_GAME

    def red_light(self):
        self.update_faces()

        if self.time_for_next_state < datetime.now() and self.players_playing:
            self.to_green_light_state()
        else:
            check_player_movement(self.cap, self.players_playing, self.players_lost)
            check_player_winning(self.players_playing, self.players_won)

        if not self.players_playing:
            self.state = GameState.END_GAME

    def end_game(self):
        print("🎉 Game over!")
        print(f"🏁 Winners: {[p.id for p in self.players_won.values()]}")
        print(f"❌ Eliminated: {[p.id for p in self.players_lost.values()]}")
        print("Press 'q' to exit...")
        
        while True:
            key = input()
            if key.lower() == 'q':
                break

    def to_green_light_state(self) -> None:
        self.set_time_for_next_state_green()
        self.state = GameState.GREEN_LIGHT
        print(f"🎮 Transitioned to state: {self.state}")

    def to_red_light_state(self) -> None:
        self.set_time_for_next_state_red()
        self.state = GameState.RED_LIGHT
        print(f"🎮 Transitioned to state: {self.state}")

    def set_time_for_next_state_green(self) -> None:
        duration = random() * GREEN_LIGHT_TIME_RANGE_SECONDS + GREEN_LIGHT_TIME_MIN_SECONDS
        self.time_for_next_state = datetime.now() + timedelta(seconds=duration)

    def set_time_for_next_state_red(self) -> None:
        self.time_for_next_state = datetime.now() + timedelta(seconds=RED_LIGHT_TIME_SECONDS)


def main() -> None:
    ui = UI(window_name="Live Feed")  # create the UI
    ui.show_start_screen()  # show start screen
    game = Game()
    game.ui = ui
    game.run()


if __name__ == '__main__':
    main()
