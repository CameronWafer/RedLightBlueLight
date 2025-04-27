import cv2
import numpy as np
import time
from motion import detect_motion
from Tracker import (track_players)
from UI import UI
import math

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))
gray_float = np.float32(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0))

mask = np.zeros((480, 640), dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (540, 380), 255, -1)

# show start screen
ui = UI("red light green light")
ui.show_start_screen()

active_players = []
game_running = True
green_light = True
start_time = time.time()

while game_running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    elapsed = time.time() - start_time
    if elapsed > 5:
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

    # pulsing red light phase
    if not green_light:
        # compute a pulsing alpha between 0.1 and 0.3 using a sine wave
        alpha = 0.2 + 0.1 * math.sin(2 * math.pi * (time.time() - start_time))
        # create a solid red image
        overlay = np.full(frame.shape, (0, 0, 255), dtype=np.uint8)  # BGR: red
        # blend it over the frame
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # detect motion
    boxes = detect_motion(frame, gray_float, mask)
    motion_detected = len(boxes) > 0

    # track and update players
    active_players = track_players(boxes, active_players, green_light)

    now = time.time()  # initiate time

    # display player count
    text = f"active players: {sum(not p.eliminated for p in active_players)}"
    pos = (10, 470)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7

    # draw black outline
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
    # draw white fill
    cv2.putText(frame, text, pos, font, scale, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # draw player boxes (green or red depending on eliminated, hide red after 3s)
    for player in active_players:
        if player.eliminated:
            if now - player.eliminated_time > 3:
                continue  # skip drawing this eliminated player
            color = (0, 0, 255)  # red for eliminated
        else:
            color = (0, 255, 0)  # green for active

        cv2.rectangle(frame, (player.x, player.y), (player.x + player.w, player.y + player.h), color, 2)
        cv2.putText(frame, f"Player {player.id[:4]}", (player.x, player.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(ui.window_name, frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # go back to start menu
        ui.show_start_screen()
        # reset game state
        active_players = []
        green_light = True
        start_time = time.time()
        continue
    elif key == ord('p'):
        # restart game state
        active_players = []
        green_light = True
        start_time = time.time()
        continue


cap.release()
cv2.destroyAllWindows()
