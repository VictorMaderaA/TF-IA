import cv2
from face_network import get_faces, get_img_faces
from image_processor import draw_rectangle, write_text
from network import *

captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera
model = get_saved_network()

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

    for img_face in faces_img:
        predict(model, img_face)

    # for k, (x, y, w, h) in enumerate(faces_location):
    #
    #
    #     img = write_text(x, y, img, str(k))

    show_image(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
