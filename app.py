from time import sleep

from face_network import get_faces, get_img_faces
from image_processor import draw_rectangle, process_img, write_text, write_text_desc
from network import *

captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera
model = get_saved_network()
switcher = {
    0: "Heart",
    1: "Oblong",
    2: "Oval",
    3: "Round",
    4: "Square"
}

descSwitcher = {
    0: "\nMont c/marco ovalados \nMont c/marco grandes y cuadrados que no superen el marco natural de tu cara \nMont c/marco de tamaño mediano",
    1: "\nMont c/marco que sean mas anchos de abajo que arriba \nMont con marco redondeado o con mas angulos \nEvitar Mont c/marco rectangulares, estrechos de alto y en forma de pera",
    2: "\nMont que cubran tus cejas \nMont que abarquen la parte superior de tu rostro de manera amplia \nMont c/marco cuadrados o redondeados, cualquiera quedara bien",
    3: "\nMont c/marco rectangulares \nMont c/marco estrechos \nEvitar Mont relativamente pequeñas con cristales redondos.",
    4: "\nMont c/marco redondeados u ovalados \nEvitar Mont c/marco de angulos muy marcados \nMont modelo aviador o estilo Cat Eye son una buena opción"
}


def get_camera_frame():
    ret, frame = captureDevice.read()
    return frame


def show_image(image):
    cv2.imshow("App", image)


i = 0

while 1:
    img = get_camera_frame()
    if i == 0:
        faces_location = get_faces(img)
        faces_img = get_img_faces(img, faces_location)

    i += 1
    img = draw_rectangle(faces_location, img)
    for k, img_face in enumerate(faces_img):
        _ = process_img(img_face)
        _ = predict(model, _)
        try:
            face = switcher.get(_.tolist().index(1), "Invalid")
            desc = descSwitcher.get(_.tolist().index(1), "-")
        except:
            face = "Invalid"
            desc = "-"
        img = write_text(faces_location[k][0], faces_location[k][1], img, face)
        img = write_text_desc(k * 80, img, face + "\n" + desc)

    show_image(img)
    if i > 15:
        i = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
