import cv2
import numpy as np
from face_detector import FaceDetector
from face_wrapper import FaceWrapper
import logger 
import os 

# CONFIG: replace video URI here
VIDEO_URI = 0
FACE_SIZE = (512, 512)

TARGET_SIZE = (128, 128)

# utils
def area(rect):
    return rect[2] * rect[3]

def main():

    frame_count = 0
    logger.debug("Loading saved model ...")
    face = FaceWrapper()
    face.load('sample')
    
    detector = FaceDetector()
    cap = cv2.VideoCapture(VIDEO_URI)

    print(face.classes)
    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     logger.debug("End video")
        #     break 
        
        # frame = cv2.flip(frame, 1)
        # people, rectedImage = detector.detect(frame, returnNewImage=True)
        # if len(people):
        #     biggest = None 
        #     for person in people:
        #         # get box having biggest area, and consider it the main person
        #         box = person
                
        #         # scale        
        #         region = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].copy()
        #         region = cv2.resize(region, FACE_SIZE)

        #         # predict 
        #         label = face.classes[np.argmax(face.predict([region])[0])]
                
        #         # draw label
        #         rectedImage = cv2.putText(
        #             rectedImage, # canvas
        #             label, # text
        #             (box[0], box[1] + 20), # bottom-left corner
        #             cv2.FONT_HERSHEY_SIMPLEX, # font   
        #             1, # font scale
        #             (0x0, 0xff, 0x0), 
        #             2, # thickness
        #             cv2.LINE_AA # line type
        #         )
                                
        # cv2.imshow('frame', rectedImage)
        ret, frame = cap.read()
        if not ret:
            logger.debug("End video")
            break
        
        pivotY = (frame.shape[0] - FACE_SIZE[0]) // 2
        pivotX = (frame.shape[1] - FACE_SIZE[0]) // 2

        frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1], pivotX:pivotX+FACE_SIZE[0]], 1)
        predictFrame = cv2.resize(frame, TARGET_SIZE) * 1.0/255

        # print(predictFrame)
        preds = face.predict([predictFrame])[0]

        logger.debug("Predict value:", preds)
        label = face.classes[np.argmax(preds)]
        
        rectedImage = cv2.putText(
                    frame, # canvas
                    label, # text
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, # font   
                    1, # font scale
                    (0x0, 0xff, 0x0), 
                    2, # thickness
                    cv2.LINE_AA # line type
        )
        cv2.imshow('classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()

main()