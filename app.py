from face_network import get_faces, get_img_faces
from image_processor import draw_rectangle, process_img, write_text
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

    for k, img_face in enumerate(faces_img):
        _ = process_img(img_face)
        _ = predict(model, _)
        img = write_text(faces_location[k][0], faces_location[k][1], img, str(_))

    show_image(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
