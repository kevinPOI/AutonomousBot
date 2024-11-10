import stag
import cv2
import numpy as np
import time

STAGSET = 21
def find_tags(frame, draw = False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected_corners) = stag.detectMarkers(frame, STAGSET)
    if draw:
        stag.drawDetectedMarkers(frame, corners, ids)
        stag.drawDetectedMarkers(frame, rejected_corners, border_color=(255, 0, 0))
        cv2.imshow("stag_detection", frame)
        cv2.waitKey(1)
    return corners