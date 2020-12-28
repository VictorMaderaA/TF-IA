import cv2
import os

folder = "resources"
train_path = "resources/train"
test_path = "resources/test"


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    print(len(images))
    return images


def resize_images(images):
    i = 0
    for x in images:
        output = cv2.resize(x, (100,100))
        cv2.imwrite('output/'+ str(i) + ".jpg" , output)
        i += 1


images = load_images_from_folder(folder)
resize_images(images)

print("Done")

def rescale_images(image):
    return cv2.resize(image, (100,100))

def load_data():
    x_train = []
    y_train_cat = []

    i = 0

    x = os.scandir(train_path)
    for directorio in x:
        for r, d, f in os.walk(directorio):
            for file in f:
                x_train.append(prepare_image(os.path.join(r, file)))
                y_train_cat.append(i)
        i += 1

def prepare_image(file_path):
    img = cv2.imread(file_path)
    # Proceso rescalado y blanco y negro
    return img

load_data()









