# import libraries
import cv2
import numpy as np
import bevTransform
import time
import matplotlib.pyplot as plt
from findTag import find_tags
from ultralytics import YOLO
from kalman import KalmanFilter
from controller import Controller
############################# SETTINGS #############################

SAVEVIDEO = False
SAVESUBTRACTION = False
#INPUTNAME = "gitignore/nhrl_a2.mp4"
INPUTNAME = "nhrl_tag2.mp4"
HouseModel = YOLO("house-bot-seg.pt")
TrackModel = YOLO("bev_subtraction_tracking2.pt")
skip_till_frame = 0 #skip the first N frame where match haven't begin
DT = 0.033 #30 fps
####################################################################

class Arena():
    def __init__(self) -> None:
        self.corners = []
    def onclick(self, event):
        x, y = event.xdata, event.ydata
        
        if len(self.corners) == 4:
            print("4 points already selected!")
        else:
            self.corners.append([x,y])
        print(f"Corners selected: {self.corners}")
    def get_crop(self, camera):
        ret, image = camera.read()
        image = pad_image_y(image, 50)
        fig, ax = plt.subplots()
        ax.imshow(image)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
class Robot():
    def __init__(self, filter = None) -> None:
        self.pose = np.zeros(3)
        self.vel = 0
        self.omega = 0
        self.update_time = time.time()
        self.filter = filter
def get_house_robot_seg(warped_image, model):
    results = model.predict(warped_image, show = False, verbose = False)
    mask_poly = results[0].masks.xy[0]
    return mask_poly

def track_robots_with_model(delta, model):
    results = model.predict(cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB),show = False, verbose = False )
    boxes = results[0].boxes.xywh
    if len(boxes) == 0:
        return [],0
    boxes[:,0] -= boxes[:,2]/2
    boxes[:,1] -= boxes[:,3]/2
    return boxes.cpu().numpy(), results[0].boxes.conf.cpu().numpy()

def background_filter(background, new_frame, rate):
    background = (background * (1-rate) + new_frame*rate).astype(np.uint8)
    return background
def get_contour_centers(contours):
    centers = []
    for x,y,w,h in contours:
        c = np.array([x + w/2, y + h/2])
        centers.append(c)
    return centers
def fuse(img1, img2):
    mode = 'weight'
    if mode == 'weight':
        w1 = 0.7
        img = (w1 * img1 + img2 * (1- w1)) // 2
    return img.astype(np.uint8)
def pad_image_y(image, y_pad):
	padding = np.zeros([y_pad, image.shape[1], image.shape[2]], dtype=np.uint8)#row, column
	image = np.vstack([image, padding])
	return image

def track_robots(warped_frame, background, us, opp, out_subtract = None):
    delta = cv2.cvtColor(cv2.subtract(background, warped_frame), cv2.COLOR_RGB2GRAY)
    delta_rgb = cv2.subtract(background, warped_frame)
    fused = fuse(delta, foreground_mask.copy())
    house_robot_seg_poly = get_house_robot_seg(warped_frame, HouseModel)
    
    house_robot_mask = np.zeros(warped_frame.shape[:2])
    cv2.fillPoly(house_robot_mask, [house_robot_seg_poly.astype(np.int32)], 255)
    delta[house_robot_mask == 255] = 0

    # if SAVESUBTRACTION:
    #     out_subtract.write(cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB))
    #out_subtract.write(warped_frame)
    # masked_warped_frame = warped_frame.copy()
    
    # cv2.imshow("masked", masked_warped_frame)
    # cv2.imshow("fused", fused)
    cv2.imshow("delta", delta)
    # cv2.imshow("delta_rgb", delta_rgb)
    #morphed = cv2.morphologyEx(delta_thresh, cv2.MORPH_OPEN, kernel)
    # morphed = cv2.threshold(delta, 100, 255,cv2.THRESH_BINARY)
    # morphed = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel)
    
    # cv2.imshow("morphed", morphed)
     # find contours 
    #contours, hier = cv2.findContours(morphed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, conf = track_robots_with_model(delta, TrackModel)
    # check every contour if are exceed certain value draw bounding boxes
    contour_list = []
    if len(contours) > 0:
        contours = contours[conf > 0.2]
        coef_areas = (contours[:,2]*contours[:,3]) * (2 + conf)
        contours = contours[np.argsort(-coef_areas)]
        for contour in contours:
            # if area exceed certain value then draw bounding boxes
            (x,y,w,h) = contour.astype(int)
            if (w * h < 25000):
                contour_list.append([x,y,w,h])
                if len(contour_list) >= 2:
                    break
    plot_two = True
    if plot_two:
        if len(contour_list) >= 2:
            centers = get_contour_centers(contour_list)
            d0 = np.linalg.norm(centers[0] - us.pose[:2])
            d1 = np.linalg.norm(centers[1] - us.pose[:2])
            (x0,y0,w0,h0) = contour_list[0]
            (x1,y1,w1,h1) = contour_list[1]
            if d0 < d1:
                cv2.rectangle(warped, (x0,y0), (x0+w0, y0+h0), (255, 255, 0), 2)
                cv2.rectangle(warped, (x1,y1), (x1+w1, y1+h1), (0,0,255), 2)
            else:
                cv2.rectangle(warped, (x0,y0), (x0+w0, y0+h0), (0,0,255), 2)
                cv2.rectangle(warped, (x1,y1), (x1+w1, y1+h1), (255,255,0), 2)
        elif len(contour_list) == 1:
            (x0,y0,w0,h0) = contour_list[0]
            cv2.rectangle(warped, (x0,y0), (x0+w0, y0+h0), (255, 255, 0), 2)
        else:
            pass

    center_list = []
    for con in contour_list:
        (x,y,w,h) = con
        center_list.append(np.array([int(x + w/2), int(y + h / 2)]))
    if SAVESUBTRACTION:
        out_subtract.write(cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB))
    return warped, center_list
def find_self_pose(frame):
    corners = find_tags(frame)
    print("corners: ", corners)
    if len(corners) == 0:
        return None
    if len(corners) > 1:
        print("warning: multiple tags found")
    corner = corners[0][0]
    pos = (corner[0,:] + corner[3,:]) / 2
    diffs = corner[1,:] - corner[0,:] #tag inverted on robot
    theta = np.arctan2(diffs[1], diffs[0])
    return np.append(pos, theta)

def dummy_ident():
    return np.array([0,0, -1.57])

def get_robots_pose(center_list, self_pose, us, opp):
    '''
    Arguments:
    center_list: list of bonding box centers of robots, maxmum two
    self_pose: self pose obtained from tag. None if tag not detected
    us, opp: robot objects

    The function identifies which bonding box corresponds to us and which corresponds to opponent. Theb pose info for both robots
    '''
    if self_pose is not None:
        use_tag_pose = True
    else:
        #if tag not detected, use last pose of our robot
        use_tag_pose = False
        self_pose = us.pose

    #find distance between self_pose from tag and bonding box position
    dists = []
    if len(center_list) == 0:
        return
    for c in center_list:
        dist = np.linalg.norm(c-  self_pose[:2])
        dists.append(dist)
    self_id = np.argmin(dists)
    if use_tag_pose:
        self_pose_new = np.concatenate([center_list[np.argmin(dists)], self_pose[2:]])
    else:
        self_pose_new = np.concatenate([center_list[np.argmin(dists)], self_pose[2:]])
    opponent_pose = np.concatenate([center_list[np.argmax(dists)], np.array([0])])

    curr_time = time.time()
    
    dt_us = curr_time - us.update_time
    us.vel = np.linalg.norm([self_pose_new - us.pose][:2])/dt_us
    us.omega = np.linalg.norm([self_pose_new-us.pose][2:])/dt_us

    dt_opp = curr_time - opp.update_time
    opp.vel = np.linalg.norm([self_pose_new - opp.pose][:2])/dt_opp
    opp.omega = np.linalg.norm([self_pose_new-opp.pose][2:])/dt_opp

    us.filter.predict(DT)
    us.filter.update(self_pose_new)
    us.pose = us.filter.get_state()[0:3]
    us.update_time = curr_time

    opp.filter.predict(DT)
    opp.filter.update(opponent_pose)
    opp.pose = opp.filter.get_state()[0:3]
    opp.update_time = curr_time


def draw_robots(warped_frame, us, opp):
    augment = True
    blank = np.zeros(warped_frame.shape)
    warped = warped_frame
    blank = cv2.circle(blank,us.pose[:2].astype(int), 5, (0,0,255), 2)
    blank = cv2.circle(blank,opp.pose[:2].astype(int), 5, (255,0,0), 2)

    magnitude = us.vel + 20
    arrow_end = (us.pose[:2] + magnitude * np.array([np.cos(us.pose[2]), np.sin(us.pose[2])])).astype(int)
    if augment:
        cv2.arrowedLine(warped,us.pose[:2].astype(int), arrow_end, (255,255,0), 2, tipLength= 0.3)
        warped = cv2.circle(warped,us.pose[:2].astype(int), 5, (255,255,0), 2)
        warped = cv2.circle(warped,opp.pose[:2].astype(int), 5, (0,0,255), 2)
        cv2.imshow("detection", warped)
    else:
        cv2.imshow("detection", warped)
        cv2.arrowedLine(blank,us.pose[:2].astype(int), arrow_end, (0,0,255), 2, tipLength= 0.3)
    # cv2.imshow("Subtractor", foreground_mask)
    # cv2.imshow("threshold", treshold)
    
        cv2.imshow("draw", blank) 

def draw_controls():
    pass
if __name__ == "__main__":
    KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = True) # detectShadows=True : exclude shadow areas from the objects you detected
    bg_subtractor=KNN_subtractor

    #camera = cv2.VideoCapture("sample1.mp4")
    camera = cv2.VideoCapture(INPUTNAME)

    while True:
        arena = Arena()
        arena.get_crop(camera)
        if len(arena.corners) == 4:
            pts = np.array(arena.corners)
            break
        else:
            print("not 4 corners are selected")
            arena.corners = []



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
    while True:
        
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
            controls = controller.get_controls()
            print("controls: ", controls)
            draw_robots(warped_boxed, us, opp)
            
            
            t1 = time.perf_counter() - t0
            print("computation time ", t1)
            if cv2.waitKey(1) & 0xff == 27:
                break
        else:
            break
            
    camera.release()
    if SAVEVIDEO:
        out.release()
    if SAVESUBTRACTION:
        out_subtract.release()
    cv2.destroyAllWindows()