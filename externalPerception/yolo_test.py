from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np

# model = YOLO("yolov8n.pt")
model = YOLO("robotv3_10e.pt")
camera = cv2.VideoCapture("nhrl_sample2.mp4")
for i in range(1300):
    
    ret, og_frame = camera.read()
    if(i > 5 and i % 4 == 0):
        t0 = time.perf_counter()
        results = model.predict(og_frame, show = True, verbose = False)
        print("infer time:", time.perf_counter() - t0)