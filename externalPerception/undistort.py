import numpy as np
import cv2

def undistort(img, resolution = 720):
    # fx = 1096.8
    # fy = 1094.8
    # cx = 948.5
    # cy = 505.8
    # D = np.array([-0.1148, 0.005666, -0.05829, 0.05917])
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float64)
    # resolution = 720
    if(resolution == 540):
        DIM=(960, 540)
        K=np.array([[548.3810822883976, 0.0, 473.536202532925], [0.0, 547.3527957989021, 252.94018853980614], [0.0, 0.0, 1.0]])
        D=np.array([[-0.10216739446794722], [-0.03306895677294249], [-0.006725804062137609], [0.03199234507215527]])
    elif(resolution == 720):
        DIM=(1280, 720)
        K=np.array([[670.972451127762, 0.0, 630.9167266687163], [0.0, 669.0123515671439, 341.1781284874536], [0.0, 0.0, 1.0]])
        D=np.array([[-0.04659622826428237], [-0.02670189918214887], [0.0026309245832455783], [-0.0032553262738900693]])
    else:
        assert False
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# img = cv2.imread("sample.jpg")
# undistort(img)