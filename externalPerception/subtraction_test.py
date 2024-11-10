# import libraries
import cv2
import numpy as np
import bevTransform
import time
import matplotlib.pyplot as plt
from findTag import find_tags

############################# SETTINGS #############################

SAVEVIDEO = False
INPUTNAME = "nhrl_tag2.mp4"


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
    def __init__(self) -> None:
        self.pose = np.zeros(3)
        self.vel = 0
        self.omega = 0
        self.update_time = time.time()
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

def track_robots(warped_frame, background, us, opp):
    delta = cv2.cvtColor(cv2.subtract(background, warped_frame), cv2.COLOR_RGB2GRAY)
    fused = fuse(delta, foreground_mask.copy())
    cv2.imshow("fused", fused)
    #morphed = cv2.morphologyEx(delta_thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.threshold(delta, 70, 255,cv2.THRESH_BINARY)
    morphed = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel)
    cv2.imshow("morphed", morphed)
     # find contours 
    contours, hier = cv2.findContours(morphed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # check every contour if are exceed certain value draw bounding boxes
    contour_list = []
    contour_color = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        # if area exceed certain value then draw bounding boxes
        if cv2.contourArea(contour) > 2500:
            (x,y,w,h) = cv2.boundingRect(contour)
            contour_list.append([x,y,w,h])
            avg_color = np.average(warped[y:y+h, x:x+w], axis = (0,1))
            contour_color.append(avg_color)
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

    # contour_color = np.asarray(contour_color)
    # if len(contour_color.shape) == 2:
    #     blue = np.argmin(contour_color[:,0] / np.sum(contour_color, axis = 1))
    #     light = np.argmax(contour_color[:,0]/ np.sum(contour_color, axis = 1))
    #     (x,y,w,h) = contour_list[blue]              
    #     cv2.rectangle(warped, (x,y), (x+w, y+h), (255, 255, 0), 2)
    #     (x,y,w,h) = contour_list[light]              
    #     cv2.rectangle(warped, (x,y), (x+w, y+h), (0, 0, 255), 2)
    center_list = []
    for con in contour_list:
        (x,y,w,h) = con
        center_list.append(np.array([int(x + w/2), int(y + h / 2)]))
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

def get_robots_pose(center_list, self_pose, us, opp):#return self_pose, opponent_pose given a list of tracking centers (maximum two) and 
    #both 
    if self_pose is not None:
        use_tag_pose = True
    else:
        use_tag_pose = False
        self_pose = us.pose

    dists = []
    if len(center_list) == 0:
        return
    for c in center_list:
        dist = np.linalg.norm(c-  self_pose[:2])
        dists.append(dist)
    self_id = np.argmin(dists)
    if use_tag_pose:
        self_pose_new = self_pose
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

    us.pose = self_pose_new
    opp.pose = opponent_pose
    us.update_time = curr_time
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
        cv2.arrowedLine(warped,us.pose[:2].astype(int), arrow_end, (0,0,255), 2, tipLength= 0.3)
        warped = cv2.circle(warped,us.pose[:2].astype(int), 5, (0,0,255), 2)
        warped = cv2.circle(warped,opp.pose[:2].astype(int), 5, (255,0,0), 2)
        cv2.imshow("detection", warped)
    else:
        cv2.imshow("detection", warped)
        cv2.arrowedLine(blank,us.pose[:2].astype(int), arrow_end, (0,0,255), 2, tipLength= 0.3)
    # cv2.imshow("Subtractor", foreground_mask)
    # cv2.imshow("threshold", treshold)
    
        cv2.imshow("draw", blank) 

# KNN
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = False) # detectShadows=True : exclude shadow areas from the objects you detected

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False) # exclude shadow areas from the objects you detected

# choose your subtractor
bg_subtractor=KNN_subtractor

#camera = cv2.VideoCapture("sample1.mp4")
camera = cv2.VideoCapture(INPUTNAME)
if True:
    while True:
        arena = Arena()
        arena.get_crop(camera)
        if len(arena.corners) == 4:
            pts = np.array(arena.corners)
            break
        else:
            print("not 4 corners are selected")
            arena.corners = []
else:
    pts = np.array([[350,50], [0,680], [980,50], [1275,680]], dtype = "float32")#sample1



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5));
first = True
gaussian_kernel = cv2.getGaussianKernel(3, 2)
gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T
us = Robot()
opp = Robot()

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
            if SAVEVIDEO:
                output_video = 'output_video.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video, fourcc, 30, warped.shape[0:2])
        # Every frame is used both for calculating the foreground mask and for updating the background. 
        else:
            background = background_filter(background, frame, 0.01)
        if SAVEVIDEO:
            out.write(warped)
        warped_boxed, center_list = track_robots(frame, background, us, opp)
        self_pose = find_self_pose(warped)
        # if self_pose_t is None:
        #     self_pose = us.pose #if no tag detected, use last bonding box position
        # else:
        #     self_pose = self_pose_t
        
        get_robots_pose(center_list, self_pose, us, opp)
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
cv2.destroyAllWindows()