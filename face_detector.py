import os
# from mtcnn import MTCNN
import cv2
from cv2 import CascadeClassifier
# import face_recognition


# FaceDetector class
class FaceDetector(object): 
    def __init__(self):
        """Init FaceDetector
        The FaceDetector will use pretrained CascadeClassifier model named haarcascade_frontalface
        """
        #
        # More about cascade models: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html 
        #

        # set core model
        self.core = CascadeClassifier('models/haarcascade_frontalface_default.xml')

    def detect(self, img, paint = True, returnNewImage = False):
        """ Detect faces on image (give by img) and paint green rects onto it
            Function use haarcascade_frontalface_default model
        Args:
            img (numpy.ndarray): The image, could be load with cv2.imread. Imread document on: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
            paint (bool, optional): . Defaults to True.
            returnNewImage (bool, optional): Whether or not to return new image containing painted rects. Defaults to False.

        Returns:
            Basically it returns rectangular regions containing faces, then
            If returnNewImage is true, functions also returns newImage with rects are painted on
        
        Example: 
            detector = FaceDetector()
            frame = cv2.imread('<path_to_image>')
            people, img = detector.detect(frame, returnNewImage=True)
            cv2.imshow('people in image', img)
        """

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
    people, new = detector.detect(root[:,:], returnNewImage=True)
    print(people)

    # show 
    cv2.imshow('frame', new)
    cv2.waitKey(0)