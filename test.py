import cv2
import depth_map
img = cv2.imread('faces/tinh/tinh_00000.jpg')
print(img.shape)
print(depth_map.generate_depth_image(img, 0, (128, 128)))
