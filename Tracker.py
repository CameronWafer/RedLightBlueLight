import time
import uuid
import numpy as np

class Player:
    def __init__(self, x, y, w, h):
        self.id = str(uuid.uuid4())[:8]
        self.x, self.y, self.w, self.h = x, y, w, h
        self.last_seen = time.time()
        self.eliminated = False
        self.eliminated_time = None # to help remove red box clutter (was hella crowded)

    def update(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.last_seen = time.time()

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

def distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def track_players(boxes, active_players, green_light, distance_threshold=50):
    now = time.time()

    # remove eliminated players after 3 seconds
    active_players = [
        p for p in active_players
        if not p.eliminated or (now - p.eliminated_time <= 3)
    ]

    for (x, y, w, h) in boxes:
        center = (x + w // 2, y + h // 2)

        if green_light:
            # match with existing player or create new
            matched = False
            for player in active_players:
                if not player.eliminated and distance(center, player.center()) < distance_threshold * 1.5:
                    player.update(x, y, w, h)
                    matched = True
                    break

            if not matched:
                active_players.append(Player(x, y, w, h))

        else:
            # check if motion is inside a green player's box (red light)
            for player in active_players:
                if player.eliminated:
                    continue
                if (player.x <= center[0] <= player.x + player.w) and (player.y <= center[1] <= player.y + player.h):
                    player.eliminated = True
                    player.eliminated_time = now

    return active_players


