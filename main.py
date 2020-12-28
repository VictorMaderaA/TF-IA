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


# Importar librerías
from keras.preprocessing.image import ImageDataGenerator
def load_data():
    # Reescalar las imágenes
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # Construir la red neuronal de convolución
    from keras import layers
    from keras import models
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Consultar la arquitectura de la red
    model.summary()
    model.compile(optimizer="Adam", loss="mse")
    # Ajustar el modelo utilizando un generador:
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    # Visualizar la pérdida y precisión del ajuste del modelo después del proceso de entrenamiento y validación.
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Precisión del Entrenamiento')
    plt.plot(epochs, val_acc, 'b', label='Precisión de la Validación')
    plt.title('Precisión del entrenamiento y validación')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Pérdida del entrenamiento')
    plt.plot(epochs, val_loss, 'b', label='Pérdida de la validacion')
    plt.title('Pérdida del entrenamiento y validación')
    plt.legend()
    plt.show()


def prepare_image(file_path):
    img = cv2.imread(file_path)
    # Proceso rescalado y blanco y negro
    return img

load_data()









