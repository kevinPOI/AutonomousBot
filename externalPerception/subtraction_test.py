# import libraries
import cv2
import numpy as np
import bevTransform
import time
def background_filter(background, new_frame, rate):
    background = (background * (1-rate) + new_frame*rate).astype(np.uint8)
    return background

def fuse(img1, img2):
    mode = 'weight'
    if mode == 'weight':
        w1 = 0.7
        img = (w1 * img1 + img2 * (1- w1)) // 2
    return img.astype(np.uint8)

# KNN
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = False) # detectShadows=True : exclude shadow areas from the objects you detected

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False) # exclude shadow areas from the objects you detected

# choose your subtractor
bg_subtractor=KNN_subtractor

camera = cv2.VideoCapture("sample1.mp4")
pts = np.array([[350,50], [0,680], [980,50], [1275,680]], dtype = "float32")#sample1

# camera = cv2.VideoCapture("nhrl_sample1.mp4")
#pts = np.array([[560,291], [137,396], [1009,337], [947,931]], dtype = "float32")#nhrl_sample1

# camera = cv2.VideoCapture("nhrl_sample2.mp4")
#pts = np.array([[372,419], [848,428], [1227,640], [0,609]], dtype = "float32")#nhrl_sample2


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5));
first = True
gaussian_kernel = cv2.getGaussianKernel(3, 2)
gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T


while True:
    t0 = time.perf_counter()
    ret, og_frame = camera.read()
    og_frame = bevTransform.pad_image_y(og_frame, 500)
    warped = bevTransform.four_point_transform(og_frame, pts)
    foreground_mask = bg_subtractor.apply(warped)
    frame = cv2.filter2D(warped, -1, gaussian_kernel_2d)
    if first:
        first = False
        #background = frame
        background = cv2.filter2D(warped, -1, gaussian_kernel_2d)
    # Every frame is used both for calculating the foreground mask and for updating the background. 
    else:
        background = background_filter(background, frame, 0.01)
    
    delta = cv2.cvtColor(cv2.subtract(background, frame), cv2.COLOR_RGB2GRAY)
    fused = fuse(delta, foreground_mask.copy())
    cv2.imshow("fused", fused)
    #morphed = cv2.morphologyEx(delta_thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.threshold(delta, 50, 255,cv2.THRESH_BINARY)
    morphed = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel)
    
     # find contours 
    contours, hier = cv2.findContours(morphed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # check every contour if are exceed certain value draw bounding boxes
    contour_list = []
    contour_color = []
    for contour in contours:
        # if area exceed certain value then draw bounding boxes
        if cv2.contourArea(contour) > 500:
            (x,y,w,h) = cv2.boundingRect(contour)
            contour_list.append([x,y,w,h])
            avg_color = np.average(warped[y:y+h, x:x+w], axis = (0,1))
            contour_color.append(avg_color)
            if len(contour_list) >= 2:
                break
    contour_color = np.asarray(contour_color)
    if len(contour_color.shape) == 2:
        blue = np.argmin(contour_color[:,0] / np.sum(contour_color, axis = 1))
        light = np.argmax(contour_color[:,0]/ np.sum(contour_color, axis = 1))
        (x,y,w,h) = contour_list[blue]              
        cv2.rectangle(warped, (x,y), (x+w, y+h), (255, 255, 0), 2)
        (x,y,w,h) = contour_list[light]              
        cv2.rectangle(warped, (x,y), (x+w, y+h), (0, 0, 255), 2)
    center_list = []
    for con in contour_list:
        (x,y,w,h) = con
        center_list.append([int(x + w/2), int(y + h / 2)])
    blank = np.zeros(warped.shape[:2])
    for center in center_list:
        (x,y) = center
        blank = cv2.circle(blank,center, 5, (255,255,255), 2)

    # cv2.imshow("Subtractor", foreground_mask)
    # cv2.imshow("threshold", treshold)
    cv2.imshow("detection", warped)
    cv2.imshow("draw", blank)
    #cv2.imshow("fused", delta_thresh)
    og_frame = cv2.resize(og_frame,(640,480))
    # cv2.imshow("raw", og_frame)
    # cv2.imshow("background", background)
    t1 = time.perf_counter() - t0
    print("computation time ", t1)
    if cv2.waitKey(1) & 0xff == 27:
        break
        
camera.release()
cv2.destroyAllWindows()