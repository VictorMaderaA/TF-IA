import cv2
import keras
import numpy as np
from PIL import Image


def draw_rectangle(faces, img):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


def write_text(x, y, img, text):
    position = (x, y)
    cv2.putText(
        img,  # numpy array on which text is written
        text,  # text
        position,  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        1,  # font size
        (209, 80, 0, 255),  # font color
        3)  # font stroke
    return img


def process_img(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Convert the captured frame into RGB
    # im = Image.fromarray(img, 'L')
    #
    # # Resizing into 128x128 because we trained the model with this image size.
    # im = im.resize((150, 150))
    # img_array = np.array(im)
    #
    # # Our keras model used a 4D tensor, (images x height x width x channel)
    # # So changing dimension 128x128x3 into 1x128x128x3
    # img_array = np.expand_dims(img_array, axis=0)
    #
    # # # Calling the predict method on model to predict 'me' on the image
    # # prediction = int(model.predict(img_array)[0][0])
    #
    # # img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img = img/255
    # img_array = img_array.reshape((4,1))
    # print(img_array.ndim)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img, 'L')
    img = img.resize((150, 150))
    doc = keras.preprocessing.image.img_to_array(img)  # -> numpy array
    # print(type(doc), doc.shape)
    doc = np.expand_dims(doc, axis=0)
    # print(type(doc), doc.shape)
    return doc
    # return img_array
