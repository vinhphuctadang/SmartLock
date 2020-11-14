'''
    facial recognition logic:
        if isFaceRecognized as A and face has normal depth => this is a real face of A and do appropriate stuff 
    https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/cameraCalibration/depthMap.py
    Based on the following tutorial:
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
'''
import numpy as np
import cv2
# configs
# BASE_LINE = 0.02 # in meter
# FOCAL_LENGTH = 0.05 # in metter

def getDepthMap(imgLeft, imgRight, BASE_LINE, FOCAL_LENGTH):
    # Initialize the stereo block matching object 
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)
    # Compute the disparity image
    disparity = stereo.compute(imgLeft, imgRight)
    disparity = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # In appropriate setup, dispaarity map wont give 0 (or both NEGATIVE and POSITIVE) result; but current implemenntaation can deal with non-standard setup, but not really accurate
    disparity[disparity == 0.] = 0.1 
    # convert to depth map
    depth = (BASE_LINE * FOCAL_LENGTH) / disparity
    # Normalize the depth for representation
    min = depth.min()
    max = depth.max()
    depth = np.uint8(255 * (depth - min) / (max - min))
    return depth

# Load the left and right images in gray scale
imgLeft = cv2.imread('faces/fakeLeft.jpg', 0)
imgRight = cv2.imread('faces/fakeRight.jpg', 0)

depth = getDepthMap(imgLeft, imgRight, 0.02, 0.05)
# Display the result
print(depth)
cv2.imshow('depth map', depth)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.destroyAllWindows()