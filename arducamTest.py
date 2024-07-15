#import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from track import track
import time
import ArducamDepthCamera as ac
MAX_DISTANCE = 4
def project(depth_image, viz = False):
    t0 = time.perf_counter()
    #fx, fy = 423, 423  # Focal lengths
    fx, fy = 140, 140  # Focal lengths with 3x down sample

    # Create meshgrid for image coordinates
    height, width = depth_image.shape
    cx, cy = width/2, height/2  # Principal point (center of the image)
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Example depth image (random values for illustration)
    #depth_image = np.random.uniform(0.5, 5.0, (height, width))

    # Convert depth image to 3D points
    # t1 = time.time() - t0
    X = (x - cx) * depth_image / fx
    Y = (y - cy) * depth_image / fy
    Z = depth_image
    #bev = np.vstack(X,Y)
    # bev = []
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         if(Z[i,j] < 2000):
    #             bev.append([X[i,j], Z[i,j]])
    # t1 = time.perf_counter() - t0
    bev = np.vstack((X.flatten(), Z.flatten())).T
    bev = bev[np.logical_and(bev[:,1]>5 , bev[:,1] < 2000)]
    # bev[:,1] = np.clip(bev[:,1], 0,2000)
    t1 = time.perf_counter() - t0
    bev = np.asarray(bev)
    if(viz):
        plt.scatter(bev[:,0], bev[:,1], s=1)

        plt.title('Bird\'s-Eye View')
        plt.axis('off')

    bev = bev.astype(np.int64)
    bev_img = np.zeros([6 * width, 2000])
    bev_width = bev_img.shape[0]
    bev= bev[np.logical_and(bev[:,0] > -bev_width/2 + 1, bev[:,0] < bev_width/2 - 1)]
    
    t2 = time.perf_counter() - t0
    mid = int(bev_width / 2)
    for b in bev:
        [px, py] = b
        bev_img[px + mid, py] += 1
    bev_img[int(width/2-3):int(width/2+3), 0:6] = 0
    t3 = time.perf_counter() - t0
    if(viz):
        cv2.imshow('bev', bev_img.T[::-1,:])
        plt.show()
        cv2.waitKey(0)
    print(t1, t2, t3)
    return bev_img

if __name__ == "__main__":
    cam = ac.ArducamCamera()
    if cam.open(ac.TOFConnect.CSI,0) != 0 :
        print("initialization failed")
    if cam.start(ac.TOFOutput.DEPTH) != 0 :
        print("Failed to start camera")
    cam.setControl(ac.TOFControl.RANG,MAX_DISTANCE)

    for i in range(500):
        frame =cam.requestFrame(200)
        depth_buf = frame.getDepthData()
        cam.releaseFrame(frame)

        depth_image = depth_buf[50:170,:]
        depth_image *= 1000
        depth_capped = np.clip(depth_image, None, 2000)
        box_kernel = np.ones([5,5])/25
        t0 = time.perf_counter()
        # depth_capped = cv2.filter2D(depth_capped, -1, box_kernel)
        depth_capped = depth_capped[::3,::3]
        # with open('test.npy', 'wb') as f:
        #     np.save(f, depth_capped)
        #color_image = np.asanyarray(color_frame.get_data())[200:320,:]
        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                        alpha = 0.5), cv2.COLORMAP_JET)
        plt.imshow(depth_capped, cmap='viridis')
        #cv2.imshow('rgb', color_image)
        # plt.show()
        # cv2.waitKey(0)
        # plt.colorbar()
        # plt.title('depth')
        plt.show()
        
        bev_img = project(depth_capped)
        track(bev_img)
        # cv2.imshow('rgb', color_image)
        cv2.waitKey(20)

    pipe.stop()
