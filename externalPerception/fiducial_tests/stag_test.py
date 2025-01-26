import stag
import cv2
import numpy as np
import time
# load image
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(True):
    
    ret, frame = cap.read()
    #frame = cv2.imread("frame_00032.png")
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    (corners, ids, rejected_corners) = stag.detectMarkers(gray, 21)
    print(ids)
    #print("detection took: ", time.perf_counter() - t0)
    # draw detected markers with ids

    
    stag.drawDetectedMarkers(frame, corners, ids)

    # draw rejected quads without ids with different color
    stag.drawDetectedMarkers(frame, rejected_corners, border_color=(255, 0, 0))
    t0 = time.perf_counter()
    # save resulting image
    cv2.imshow("detection", frame)
    cv2.waitKey(1)
    #print("show took: ", time.perf_counter() - t0)
