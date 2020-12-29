import cv2
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from network import build_network, store_network
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


folder = "resources"
train_path = "resources/train"
test_path = "resources/test"


# Importar librerías
def load_data():
    # Reescalar las imágenes
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        color_mode="grayscale")

    testing_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(150, 150),
        color_mode="grayscale")

    print("Train Gen")
    print(train_generator.class_indices)
    print("Test Gen")
    print(testing_generator.class_indices)

    network = build_network()

    # Ajustar el modelo utilizando un generador:
    history = network.fit_generator(
        train_generator,
        epochs=30,
        validation_data=testing_generator,
        validation_steps=50)

    store_network(network)


    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


load_data()
