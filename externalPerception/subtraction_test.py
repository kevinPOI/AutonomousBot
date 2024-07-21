# import libraries
import cv2
import numpy as np
import bevTransform


# KNN
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = False) # detectShadows=True : exclude shadow areas from the objects you detected

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False) # exclude shadow areas from the objects you detected

# choose your subtractor
bg_subtractor=KNN_subtractor

camera = cv2.VideoCapture("videoL.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5));
first = True
gaussian_kernel = cv2.getGaussianKernel(10, 5)
gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T
pts = np.array([[350,50], [0,680], [980,50], [1275,680]], dtype = "float32")    
while True:
    ret, og_frame = camera.read()
    warped = bevTransform.four_point_transform(og_frame, pts)

    if first:
        first = False
        #background = frame
        background = cv2.filter2D(warped, -1, gaussian_kernel_2d)
    # Every frame is used both for calculating the foreground mask and for updating the background. 
    
    foreground_mask = bg_subtractor.apply(warped)
    frame = cv2.filter2D(warped, -1, gaussian_kernel_2d)
    delta = cv2.cvtColor(cv2.subtract(background, frame), cv2.COLOR_RGB2GRAY)
    # threshold if it is bigger than 240 pixel is equal to 255 if smaller pixel is equal to 0
    # create binary image , it contains only white and black pixels
    ret , treshold = cv2.threshold(foreground_mask.copy(), 120, 255,cv2.THRESH_BINARY)
    ret, delta_thresh = cv2.threshold(delta, 50, 255,cv2.THRESH_BINARY)
    #  dilation expands or thickens regions of interest in an image.
    #dilated = cv2.dilate(treshold,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)),iterations = 2)
    dilated = cv2.dilate(delta_thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),iterations = 2)
    fused = (np.logical_or(treshold, delta_thresh).astype(np.uint8)*255)
    #morphed = cv2.morphologyEx(delta_thresh, cv2.MORPH_OPEN, kernel)
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
    blue = np.argmin(contour_color[:,0] / np.sum(contour_color, axis = 1))
    light = np.argmax(contour_color[:,0]/ np.sum(contour_color, axis = 1))
    (x,y,w,h) = contour_list[blue]              
    cv2.rectangle(warped, (x,y), (x+w, y+h), (255, 255, 0), 2)
    (x,y,w,h) = contour_list[light]              
    cv2.rectangle(warped, (x,y), (x+w, y+h), (0, 0, 255), 2)

    # cv2.imshow("Subtractor", foreground_mask)
    # cv2.imshow("threshold", treshold)
    cv2.imshow("detection", warped)
    #cv2.imshow("morphed", morphed)
    #cv2.imshow("fused", delta_thresh)
    og_frame = cv2.resize(og_frame,(640,480))
    cv2.imshow("raw", og_frame)
    
    if cv2.waitKey(30) & 0xff == 27:
        break
        
camera.release()
cv2.destroyAllWindows()