import cv2
import numpy as np

def boxes_overlap(box1, box2, threshold=0.3):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return False

    iou = inter_area / union_area
    return iou > threshold


def detect_motion(frame, background_model, mask=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.accumulateWeighted(gray, background_model, 0.05, mask=mask)
    adaptive_bg = cv2.convertScaleAbs(background_model)

    diff = cv2.absdiff(adaptive_bg, gray)
    masked = cv2.bitwise_and(diff, diff, mask=mask)
    _, thresh = cv2.threshold(masked, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > 2000]
    raw_boxes = [cv2.boundingRect(c) for c in filtered]

    # remove overlapping boxes
    # this is much better than preventing overlapped boxes as that just stops creation...
    #     ...of boxes as a whole
    boxes = []
    for box in raw_boxes:
        if not any(boxes_overlap(box, b) for b in boxes):
            boxes.append(box)

    return boxes
