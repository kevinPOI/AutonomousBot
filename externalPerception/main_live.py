# import libraries
import cv2
import numpy as np
import bevTransform
import time
import os
t0 = time.perf_counter()
initial_time = time.time()
import matplotlib.pyplot as plt
from undistort import undistort
from recording import create_new_recording_folder
from main import background_filter, track_robots, find_self_pose, get_robots_pose, draw_robots
from main import Arena
from main import Robot
from findTag import find_tags
from ultralytics import YOLO
from kalman import KalmanFilter
from controller import Controller
from radio import Radio
print(time.perf_counter() - t0)
############################# SETTINGS #############################

CAMERA = 8
RECORD = False
SAVEVIDEO = False
SAVESUBTRACTION = False
undistort_camera = False
skip_till_frame = 0 #skip the first N frame where match haven't begin
HouseModel = YOLO("house-bot-seg.pt")
TrackModel = YOLO("bev_subtraction_tracking2.pt")
DT = 0.033 #30 fps
sim_target = np.array([200,200,0])
####################################################################
def save_frame(frame, frame_counter, folder_path):
    frame_filename = os.path.join(folder_path, f"frame_{frame_counter:05d}.png")
    cv2.imwrite(frame_filename, frame)

if __name__ == "__main__":
    print(time.perf_counter() - t0)
    # KNN
    KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = True) # detectShadows=True : exclude shadow areas from the objects you detected
    bg_subtractor=KNN_subtractor
    camera = cv2.VideoCapture(CAMERA)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #camera = cv2.VideoCapture("sample1.mp4")

    while True:
        arena = Arena()
        arena.get_crop(camera)
        if len(arena.corners) == 4:
            pts = np.array(arena.corners)
            break
        else:
            print("not 4 corners are selected")
            arena.corners = []


    #camera = cv2.VideoCapture("sample1.mp4")
    # camera = cv2.VideoCapture(CAMERA)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # if True:
    #     while True:
    #         arena = Arena()
    #         arena.get_crop(camera)
    #         if len(arena.corners) == 4:
    #             pts = np.array(arena.corners)
    #             break
    #         else:
    #             print("not 4 corners are selected")
    #             arena.corners = []
            

    else:
        pts = np.array([[350,50], [0,680], [980,50], [1275,680]], dtype = "float32")#sample1
    if RECORD:
        folder_name = create_new_recording_folder()

    timestamps = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5));
    first = True
    gaussian_kernel = cv2.getGaussianKernel(3, 2)
    gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T


    frame_count = 0
    initial_state = [0, 0, 0, 0, 0, 0]
    process_noise = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    measurement_noise = np.diag([0.05, 0.05, 0.05])
    kf_us = KalmanFilter(initial_state, process_noise, measurement_noise)
    us = Robot(filter=kf_us)
    kf_opp = KalmanFilter(initial_state, process_noise, measurement_noise)
    opp = Robot(kf_opp)
    controller = Controller(us, opp)
    radio = Radio()

    for i in range(int(10e3)):
        t0 = time.perf_counter()
        ret, og_frame = camera.read()
        if(ret):
            og_frame = bevTransform.pad_image_y(og_frame, 500)
            warped = bevTransform.four_point_transform(og_frame, pts)
            foreground_mask = bg_subtractor.apply(warped)
            #frame = cv2.filter2D(warped, -1, gaussian_kernel_2d)
            frame = warped
            if first:
                first = False
                #background = frame
                background = cv2.filter2D(warped, -1, gaussian_kernel_2d)
                controller.frame_w, controller.frame_h = warped.shape[:2]
                if SAVEVIDEO:
                    output_video = 'output_video.mp4'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video, fourcc, 30, warped.shape[0:2])
                if SAVESUBTRACTION:
                    output_video_subtract = 'subtraction_output_video.avi'
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out_subtract = cv2.VideoWriter(output_video_subtract, fourcc, 30, warped.shape[0:2])

            # Every frame is used both for calculating the foreground mask and for updating the background. 
            else:
                frame_count += 1
                background = background_filter(background, frame, 0.01)
            if frame_count < skip_till_frame:
                continue 
            if SAVEVIDEO:
                out.write(warped)
            warped_boxed, center_list = track_robots(frame, background, us, opp)
            self_pose = find_self_pose(warped)
            # if self_pose_t is None:
            #     self_pose = us.pose #if no tag detected, use last bonding box position
            # else:
            #     self_pose = self_pose_t
            
            get_robots_pose(center_list, self_pose, us, opp)
            opp.pose = sim_target
            controls = controller.get_controls()
            target = radio.send_control(controls)
            if not (target is None):
                sim_target = target
            
            print("controls: ", controls)
            draw_robots(warped_boxed, us, opp)
            
            
            t1 = time.perf_counter() - t0
            print("computation time ", t1)
            if cv2.waitKey(1) & 0xff == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()