from PIL import Image, ImageDraw
import face_recognition
import cv2
# Load the jpg file into a numpy array
image = face_recognition.load_image_file('faces/tinh/tinh_00000.jpg')

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)
face_bounding_boxes = face_recognition.face_locations(image)
top, right, bottom, left = face_bounding_boxes[0]
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(
            facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

d.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))

# Show the picture
pil_image.show()
