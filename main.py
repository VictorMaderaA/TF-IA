import math

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from network import build_network, store_network, get_saved_network
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

folder = "resources"
train_path = "resources/train"
test_path = "resources/test"


# Mostramos Metricas de una historia
def show_metrics(history):
    print(history.history)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# Cargamos Data
def load_data():
    # Reescalar las imágenes
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(32, 32),
        color_mode="grayscale", )

    testing_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(32, 32),
        color_mode="grayscale")

    print("Train Gen")
    print(train_generator.class_indices)
    print("Test Gen")
    print(testing_generator.class_indices)

    return train_generator, testing_generator


# Construimos y entrenamos la red
def train():
    network = build_network()

    train_data, test_data = load_data()
    # Ajustamos el modelo utilizando un generador:
    history = network.fit_generator(
        train_data,
        epochs=100,
        validation_data=test_data,
        steps_per_epoch=75)

    store_network(network)

    show_metrics(history)

    predict(network)


# Predecimos sobre una red
def predict(network):
    train_data, test_data = load_data()
    predicted_classes = network.predict_classes(test_data)
    correct = np.nonzero(predicted_classes == test_data.labels)[0]
    incorrect = np.nonzero(predicted_classes != test_data.labels)[0]
    print("**********MATRIZ DE CONFUSION**********")
    print(confusion_matrix(test_data.labels, predicted_classes))
    print(len(correct), "Clasificados correctamente")
    print(len(incorrect), " Clasificados incorrectamente")
    print()
    print("**********REPORTE DE CLASIFICACION**********")
    print(classification_report(test_data.labels, predicted_classes))


train()
