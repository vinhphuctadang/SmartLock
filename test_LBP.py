import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from scipy.spatial import distance


def LBP(image, numPoints, radius, eps=1e-7):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="var")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return np.array([hist])


img1 = cv2.imread('faces/tinh/tinh_00000.jpg')
lbp1 = LBP(img1, 12, 1.5)[0]
print(lbp1)

# img1 = cv2.imread('/Users/dcongtinh/Downloads/IMG_4940.jpg')
# lbp2 = LBP(img1, 12, 1.5)[0]
# print(lbp2)

# print(distance.euclidean(lbp1, lbp2))
# # # gaussian_numbers = np.random.normal(size=1000)
# # # plt.hist(gaussian_numbers)
# # plt.hist(lbp, bins='auto')
# # plt.title("LBPV Histogram")
# # plt.xlabel("Value")
# # plt.ylabel("Frequency")
# # plt.show()
