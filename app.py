from time import sleep

from face_network import get_faces, get_img_faces
from image_processor import draw_rectangle, process_img, write_text
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


def get_camera_frame():
    ret, frame = captureDevice.read()
    return frame


def show_image(image):
    cv2.imshow("App", image)


while 1:
    img = get_camera_frame()
    faces_location = get_faces(img)
    img = draw_rectangle(faces_location, img)
    faces_img = get_img_faces(img, faces_location)

    for k, img_face in enumerate(faces_img):
        _ = process_img(img_face)
        _ = predict(model, _)
        try:
            face = switcher.get(_.tolist().index(1), "Invalid")
        except:
            face = "Invalid"
        img = write_text(faces_location[k][0], faces_location[k][1], img, face)

    show_image(img)
    sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
