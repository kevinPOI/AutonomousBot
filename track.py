import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

def log_scale_normalize(matrix, epsilon=1e-9):
    # Take the logarithm of all elements, adding epsilon to avoid log(0)
    log_matrix = np.log(matrix + epsilon)
    #log_matrix = matrix
    # Normalize the log-transformed matrix to the range [0, 1]
    min_val = np.min(log_matrix)
    max_val = np.max(log_matrix)
    
    normalized_matrix = (log_matrix - min_val) / (max_val - min_val)
    
    return normalized_matrix

def xy_to_rc(xy):
    # widht, height = matrix.shape[:2]
    return [xy[1], xy[0]]
def slope(x1,y1,x2,y2):
    ###finding slope
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

def drawLine(image,x1,y1,x2,y2, color = 0, width = 50):

    m=slope(x1,y1,x2,y2)
    h,w=image.shape[:2]
    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, width)
    # plt.matshow(image)
    # plt.matshow(image.T[::-1,:])
    # plt.show()
    # pass




def find_intersections(image_width, image_height, theta, point):
    image_width -= 3
    image_height -= 3
    x0, y0 = point
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    tan_theta = np.tan(theta)

    intersections = []

    # Intersection with left edge (x = 0)
    if cos_theta != 0:
        t = -x0 / cos_theta
        y = y0 + t * sin_theta
        if 0 <= y <= image_height:
            intersections.append((0, y))

    # Intersection with right edge (x = image_width)
    if cos_theta != 0:
        t = (image_width - x0) / cos_theta
        y = y0 + t * sin_theta
        if 0 <= y <= image_height:
            intersections.append((image_width, y))

    # Intersection with top edge (y = 0)
    if sin_theta != 0:
        t = -y0 / sin_theta
        x = x0 + t * cos_theta
        if 0 <= x <= image_width:
            intersections.append((x, 0))

    # Intersection with bottom edge (y = image_height)
    if sin_theta != 0:
        t = (image_height - y0) / sin_theta
        x = x0 + t * cos_theta
        if 0 <= x <= image_width:
            intersections.append((x, image_height))

    return np.int32(np.asarray(intersections))



def track(bev_image, viz=False):
    t0 = time.perf_counter()
    box_kernel = np.ones([5,5])/25
    
    bev_image = cv2.filter2D(bev_image, -1, box_kernel)
    bev_image = bev_image[::3,::3]
    #bev_image = cv2.resize(bev_image, np.int32(np.asarray(bev_image.shape)/2))
    log_bev = log_scale_normalize(bev_image)
    grayscale_matrix = (log_bev * 255).astype(np.uint8)
    colored_matrix_original = cv2.cvtColor(grayscale_matrix, cv2.COLOR_GRAY2BGR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    close = cv2.morphologyEx(grayscale_matrix, cv2.MORPH_CLOSE, kernel, iterations=3) 
    
    # gaussian_kernel = cv2.getGaussianKernel(55, 30)
    # gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T
    # close = cv2.filter2D(close, -1, gaussian_kernel_2d)
    # close = cv2.erode(close, kernel1, iterations=1) 
    # cv2.imshow('blured', np.transpose(close, (1, 0))[::-1,:])
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(close,1,6 * np.pi/180,50,100,10)
    #reconstruct = np.zeros_like(close)
    # cv2.imshow('grayscale', grayscale_matrix.T[::-1,:])
    close = cv2.cvtColor(close, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('close', close.T[::-1,:])
    grayscale_matrix += 5
    lines_mat = np.asarray(lines).squeeze()
    #lengths = np.linalg.norm([lines_mat[:,2] - lines_mat[:,0], lines_mat[:,3] - lines_mat[:,1]], axis = 0)
    thetas = np.arctan2((lines_mat[:,2] - lines_mat[:,0]), (lines_mat[:,3] - lines_mat[:,1]))
    thetas = thetas + (thetas < 0) * 2*np.pi
    th0 = thetas[0]
    th1 = -1
    th1_i = -1
    # t_rand = [] buffer for extranuous lines that aren't walls: not used
    for i in range(len(thetas)):
        th = thetas[i]
        if abs(th - th0) > 0.2:
            if abs(1.57 - abs(th - th0)) < 0.2:
                t1 = th
                th1_i = i
                break
            # else:
            #     t_rand.append(i)
    drawLine(grayscale_matrix, lines_mat[0,0], lines_mat[0,1], lines_mat[0,2], lines_mat[0,3])
    if(th1_i != -1):
        drawLine(grayscale_matrix, lines_mat[th1_i,0], lines_mat[th1_i,1], lines_mat[th1_i,2], lines_mat[th1_i,3])
    #cv2.line(grayscale_matrix,xy_to_rc(t0_ends[0]),xy_to_rc(t0_ends[1]),255,60)
    # cv2.line(grayscale_matrix,xy_to_rc(t1_ends[0]),xy_to_rc(t1_ends[1]),255,60)
    # cv2.line(grayscale_matrix,t0_ends[0],t0_ends[1],255,60)
    # cv2.line(grayscale_matrix,xy_to_rc(t1_ends[0]),xy_to_rc(t1_ends[1]),255,60)
    # cv2.line(grayscale_matrix,(50,10),(100,200),255,60)
    # cv2.line(grayscale_matrix,(350,660),(420,490),255,60)
    # for line in [lines[0], lines[t1_i]]:
    #     for x1,y1,x2,y2 in line:
    #         length = np.linalg.norm([x2-x1, y2-y1])
    #         #dist_to_sensor = np.linalg.norm([(x2+x1)/2, (y2+y1)/2])
    #         cv2.line(grayscale_matrix,(x1,y1),(x2,y2),0,60)
    #print("failed to find walls!")
                #cv2.line(reconstruct, (x1,y1),(x2,y2),255,60)
    #reconstruct = cv2.erode(reconstruct, kernel1, iterations=1) 
    #lines2 = cv2.HoughLinesP(reconstruct,1,6 * np.pi/180,100,150,30)
    # cv2.imshow('close', np.transpose(close, (1, 0, 2))[::-1,:,:])
    # cv2.imshow('grayscale', grayscale_matrix.T[::-1,:])
    # cv2.waitKey(0)

    gaussian_kernel = cv2.getGaussianKernel(50, 40)
    gaussian_kernel_2d = gaussian_kernel * gaussian_kernel.T
    conved_grayscale = cv2.filter2D(grayscale_matrix, -1, gaussian_kernel_2d)
    
    

    target = np.unravel_index(np.argmax(conved_grayscale), conved_grayscale.shape)
    confidence = conved_grayscale[target]
    threshold = max(40, 75 - target[1] / 20)
    t4 = time.perf_counter() - t0
    print("track takes: ", t4)
    if confidence > threshold:
        print("Found with confidence: ", confidence - threshold)
    else:
        print("Not found with confidence: ", confidence - threshold)
    #plt.matshow(conved_grayscale.T[::-1,:])
    
    cv2.circle(colored_matrix_original,xy_to_rc(target),25, (0,0,255),2)
    cv2.line(colored_matrix_original, xy_to_rc([int(colored_matrix_original.shape[0]/2), 0]), xy_to_rc(target), (20,20,220), 2)
    drawLine(colored_matrix_original, lines_mat[0,0], lines_mat[0,1], lines_mat[0,2], lines_mat[0,3], color=(120,120,20), width = 4)
    drawLine(colored_matrix_original, lines_mat[th1_i,0], lines_mat[th1_i,1], lines_mat[th1_i,2], lines_mat[th1_i,3], color=(120,120,20), width = 4)
    cv2.imshow('target', np.transpose(colored_matrix_original, (1, 0, 2))[::-1,:,:])
    #cv2.imshow('reconstruct', np.transpose(reconstruct, (1, 0))[::-1,:])
    #cv2.waitKey(10)
    # plt.show()
    # cv2.waitKey(0)
    
    pass
if __name__ == "__main__":
    from realsenceTest import project
    with open('test.npy', 'rb') as f:
        img = np.load(f)
    t0 = time.time()
    bev_image = project(img)
    t1 = time.time()
    print("project takes: ", t1 - t0)
    for i in range(50):
        track(bev_image)
    

