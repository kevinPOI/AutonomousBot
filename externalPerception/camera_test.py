# import libraries
import cv2
import numpy as np
import bevTransform
import time
import matplotlib.pyplot as plt
from undistort import undistort
############################# SETTINGS #############################

CAMERA = 0


###################################################################

camera = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# camera = cv2.VideoCapture(CAMERA)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5));
first = True
gaussian_kernel = cv2.getGaussianKernel(3, 2)
gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T
us = Robot()
opp = Robot()

for i in range(100000):
    t = time.perf_counter()
    ret, og_frame = camera.read()
    ud = undistort(og_frame)
    print(og_frame.shape)
    print(time.perf_counter() - t)
    if ret:
        if i % 2 == 0:
            cv2.imshow("frame", og_frame)
            cv2.imshow("ud", ud)
            cv2.waitKey(1)
camera.release()
cv2.destroyAllWindows()