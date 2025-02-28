import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	x_sort = pts[pts[:,0].argsort()]
	if x_sort[0,1] < x_sort[1,1]:
		tl = x_sort[0,:]
		bl = x_sort[1,:]
	else:
		tl = x_sort[1,:]
		bl = x_sort[0,:]
	if x_sort[2,1] < x_sort[3,1]:
		tr = x_sort[2,:]
		br = x_sort[3,:]
	else:
		tr = x_sort[3,:]
		br = x_sort[2,:]

	rect = np.asarray([tl, tr, br, bl], dtype=np.float32)
	return rect

def pad_image_y(image, y_pad):
	padding = np.zeros([y_pad, 
	
	image.shape[1], image.shape[2]], dtype=np.uint8)#row, column
	image = np.vstack([image, padding])
	return image
def transform_points(points, pts):
    """
    Transforms an array of points using the perspective warp defined by four points.
    
    Args:
        points (numpy array): An Nx2 array of points to transform.
        pts (numpy array): Four points defining the transformation.

    Returns:
        numpy array: An Nx2 array of transformed points.
    """
    # Obtain the consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the dimensions of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Transform the points
    input_points = np.array(points, dtype="float32").reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(input_points, M)

    # Extract the transformed points
    transformed_points = transformed_points.reshape(-1, 2)

    # Scale the points to the final square dimensions
    side_length = min(maxHeight, maxWidth)
    scale_x = side_length / maxWidth
    scale_y = side_length / maxHeight

    # Apply scaling
    transformed_points[:, 0] *= scale_x
    transformed_points[:, 1] *= scale_y

    return transformed_points

def transform_single_point(x, y, pts):
    """
    Transforms a single point (x, y) using the perspective warp defined by four points.
    
    Args:
        x (float): x-coordinate of the point to transform.
        y (float): y-coordinate of the point to transform.
        pts (numpy array): Four points defining the transformation.

    Returns:
        (float, float): Transformed coordinates of the point.
    """
    # Obtain the consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the dimensions of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Transform the point
    input_point = np.array([[x, y]], dtype="float32")
    input_point = np.array([input_point])
    transformed_point = cv2.perspectiveTransform(input_point, M)

    # Extract the coordinates of the transformed point
    tx, ty = transformed_point[0, 0]

    # Scale the point to the final square dimensions
    side_length = min(maxHeight, maxWidth)
    scale_x = side_length / maxWidth
    scale_y = side_length / maxHeight

    # Apply scaling
    tx_scaled = tx * scale_x
    ty_scaled = ty * scale_y

    return tx_scaled, ty_scaled

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	side_length = min(maxHeight, maxWidth)
	warped = cv2.resize(warped, (side_length, side_length))
	return warped
def onclick(event):
    x, y = event.xdata, event.ydata
    print(f"Coordinates: ({x}, {y})")

if __name__ == "__main__":
	calibrate = True
	camera = cv2.VideoCapture("nhrl_sample2.mp4")
	ret, image = camera.read()
	image = pad_image_y(image, 50)
	if calibrate:
		fig, ax = plt.subplots()
		ax.imshow(image)
		cid = fig.canvas.mpl_connect('button_press_event', onclick)
		plt.show()

	if False:
		image = cv2.imread("sample.png")#1255x655
	
	#pts = np.array([[350,50], [0,630], [950,50], [1250,630]], dtype = "float32")
	#pts = np.array([[560,291], [137,396], [1009,337], [947,931]], dtype = "float32")#nhrl_sample1
	pts = np.array([[372,419], [848,428], [1227,640], [0,609]], dtype = "float32")#nhrl_sample2
	warped = four_point_transform(image, pts)
	for pt in pts:
		cv2.circle(image,pt.astype(int),5, (0,0,255),2)
	# show the original and warped images
	cv2.imshow("Original", image)
	cv2.imshow("Warped", warped)
	cv2.waitKey(0)