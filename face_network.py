import cv2
import numpy as np

classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')


def get_faces(colored_img):
    img_copy = np.copy(colored_img)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for key, (x, y, w, h) in enumerate(faces):
        exp = int(w * 0.60)
        hexp = int(exp / 2)
        faces[key] = (x - hexp, y - hexp, w + exp, h + exp)
    return faces


def get_img_faces(img, faces):
    img_faces = []
    for (x, y, w, h) in faces:
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        try:
            img_faces.append(img[y:y + h, x:x + w])
        except:
            print("Error - Face Outside Area")
    return img_faces
