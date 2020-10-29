import cv2
import os

folder = "resources"


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



