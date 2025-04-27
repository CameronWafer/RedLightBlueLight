import cv2
import numpy as np
import time
from motion import detect_motion
from Tracker import track_players
from UI import UI

def main():
    # initialize camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("could not read from camera")
        return

    # warm up and create background model
    frame = cv2.resize(frame, (640, 480))
    gray_float = np.float32(
        cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (7, 7), 0
        )
    )

    # define region of interest mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (540, 380), 255, -1)

    # show start screen
    ui = UI("red light green light game")
    ui.show_start_screen()

    # game state variables
    active_players = []
    green_light = True
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        # toggle light every 5 seconds
        if time.time() - start_time > 5:
            green_light = not green_light
            start_time = time.time()

        # show current game state as a traffic light
        h, w = frame.shape[:2]
        light_w, light_h = 60, 140
        margin = 20
        x0 = w - light_w - margin
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
        if green_light:
            cv2.circle(frame, (cx, red_y), radius, dark, -1)  # red off
            cv2.circle(frame, (cx, green_y), radius, (0, 255, 0), -1)  # green on
        else:
            cv2.circle(frame, (cx, red_y), radius, (0, 0, 255), -1)  # red on
            cv2.circle(frame, (cx, green_y), radius, dark, -1)  # green off

        # detect motion and track players
        boxes = detect_motion(frame, gray_float, mask)
        active_players = track_players(boxes, active_players, green_light)
        now = time.time()

        # display player count
        count = sum(not p.eliminated for p in active_players)
        cv2.putText(
            frame,
            f"Active Players: {count}",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # draw player boxes
        for player in active_players:
            if player.eliminated and (now - player.eliminated_time) > 3:
                continue
            color = (0, 0, 255) if player.eliminated else (0, 255, 0)
            cv2.rectangle(
                frame,
                (player.x, player.y),
                (player.x + player.w, player.y + player.h),
                color,
                2
            )
            cv2.putText(
                frame,
                f"Player {player.id[:4]}",
                (player.x, player.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # handle display and user input
        cv2.imshow("red light green light game", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # restart to start screen
            ui.show_start_screen()
            active_players = []
            green_light = True
            start_time = time.time()
            continue

    cap.release()
    ui.destroy()

if __name__ == "__main__":
    main()
