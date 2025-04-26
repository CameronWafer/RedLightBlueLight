import cv2
import numpy as np
import time
from motion import detect_motion
from Tracker import (track_players)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))
gray_float = np.float32(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0))

mask = np.zeros((480, 640), dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (540, 380), 255, -1)

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

    # show current game state
    state_text = "Green Light" if green_light else "Red Light"
    cv2.putText(frame, state_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 0) if green_light else (0, 0, 255), 3)

    # detect motion
    boxes = detect_motion(frame, gray_float, mask)
    motion_detected = len(boxes) > 0

    # track and update players
    active_players = track_players(boxes, active_players, green_light)

    now = time.time() #initiate time

    # display player count
    cv2.putText(frame, f"Active Players: {sum(not p.eliminated for p in active_players)}",
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

    cv2.imshow("Red Light Green Light Game", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
