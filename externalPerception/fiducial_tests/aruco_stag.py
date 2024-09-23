# compare aruco and stag
import cv2
import cv2.aruco as aruco
import stag
import time
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the dictionary we are using to detect the ArUco markers
# DICT_6X6_250 is one example; you can choose a different one if you need to

# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

parameters = cv2.aruco.DetectorParameters()
# parameters.polygonalApproxAccuracyRate = 0.1
parameters.maxErroneousBitsInBorderRate = 0.9
parameters.errorCorrectionRate = 0.5
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

parameters2 = cv2.aruco.DetectorParameters()

detector2 = cv2.aruco.ArucoDetector(aruco_dict, parameters2)
detect_rate = 0
detect_rate2 = 0

aruco_detected_num = 0
aruco2_detected_num = 0
stag_detected_num = 0


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale (ArUco detection works better on a grayscale image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # stag
    t0 = time.perf_counter()
    (corners_stag, ids_stag, rejected_corners) = stag.detectMarkers(gray, 21)

    # Detect markers
    corners_aruco, ids_aruco, rejectedImgPoints = detector.detectMarkers(gray)
    corners2_aruco, ids2_aruco, rejectedImgPoints2 = detector2.detectMarkers(gray)

    # Draw detected markers on the frame
    if ids_aruco is not None:
        gray = aruco.drawDetectedMarkers(gray, corners_aruco, ids_aruco)
        detect_rate = len(ids_aruco)
    else:
        detect_rate = 0

    if ids2_aruco is not None:
        detect_rate2 = len(ids_aruco)
    else:
        detect_rate2 = 0
    
    aruco_detected_num+= detect_rate
    aruco2_detected_num+= detect_rate2
    stag_detected_num+= len(ids_stag)



    # not drawing on the same frame ???

    #stag
    stag.drawDetectedMarkers(gray, corners_stag, ids_stag)
    stag.drawDetectedMarkers(gray, rejected_corners, border_color=(255, 0, 0))


    # Display the resulting frame
    cv2.imshow('Detection', gray)
    print(detect_rate, " : ", detect_rate2)
    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imshow('Aruco Marker Detection', frame)
        cv2.waitKey(0)
        break
    
    print("stag corners: ", corners_stag)
    # compare the two , see which detects more of the time
    print("aruco detect rate: ", aruco_detected_num)
    print("aruco2 detect rate: ", aruco2_detected_num)
    print("stag detect rate: ", stag_detected_num)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
