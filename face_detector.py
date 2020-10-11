import os
from mtcnn import MTCNN
import cv2
from cv2 import CascadeClassifier

class FaceDetector(object): 
    
    def __init__(self):
        # set core model
        self.core = CascadeClassifier('models/haarcascade_frontalface_default.xml')

    def detect(self, img, paint = True, returnNewImage = False):
        # detect faces
        people = [ list(x) for x in self.core.detectMultiScale(img) ]
        # paint rects onto images
        if paint:
            # clone the image first
            if returnNewImage:
                img = img.copy()

            for person in people:
                rect = person
                cv2.rectangle(
                    img,
                    # top-left
                    (rect[0], rect[1]), 
                    # bottom-right
                    (rect[0] + rect[2], rect[1] + rect[3]), 
                    # color as RGB
                    (0x0, 0xff, 0x0), 
                    # thick ness
                    2 
                )

            if returnNewImage:
                return people, img

        return people

if __name__=='__main__':
    # read image
    root = cv2.imread("./data/sample.jpg")

    # new detector to get people
    detector = FaceDetector()
    people, new = detector.detect(root, returnNewImage=True)
    print(people)

    # show 
    cv2.imshow('frame', new)
    cv2.waitKey(0)