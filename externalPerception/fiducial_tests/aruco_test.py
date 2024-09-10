import cv2
import cv2.aruco as aruco

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the dictionary we are using to detect the ArUco markers
# DICT_6X6_250 is one example; you can choose a different one if you need to

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
# parameters.polygonalApproxAccuracyRate = 0.1
parameters.maxErroneousBitsInBorderRate = 0.9
parameters.errorCorrectionRate = 0.5
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

parameters2 = cv2.aruco.DetectorParameters()

detector2 = cv2.aruco.ArucoDetector(aruco_dict, parameters2)
detect_rate = 0
detect_rate2 = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale (ArUco detection works better on a grayscale image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    corners2, ids2, rejectedImgPoints2 = detector2.detectMarkers(gray)

    # Draw detected markers on the frame
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        detect_rate = len(ids)
    else:
        detect_rate = 0

    if ids2 is not None:
        detect_rate2 = len(ids)
    else:
        detect_rate2 = 0

    # Display the resulting frame
    cv2.imshow('Aruco Marker Detection', frame)
    print(detect_rate, " : ", detect_rate2)
    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imshow('Aruco Marker Detection', frame)
        cv2.waitKey(0)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
