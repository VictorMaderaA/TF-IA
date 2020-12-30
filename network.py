import cv2
from keras import models
from keras import layers
from keras.models import model_from_json
import numpy as np

globals()['NET_MODEL'] = "network_model.json"
globals()['NET_WEIGHTS'] = "network_weights.h5"


def build_network():
    network = models.Sequential()

    # Capa entrada
    network.add(layers.Conv2D(10, (5, 5), activation='relu', input_shape=(150, 150, 1)))
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Conv2D(20, (5, 5), activation='relu'))
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Conv2D(10, (5, 5), activation='relu'))
    network.add(layers.MaxPooling2D((2, 2)))

    # Hacemos un flatten para poder usar una red fully connected
    network.add(layers.Flatten())
    network.add(layers.Dense(100, activation='relu'))
    network.add(layers.Dense(100, activation='relu'))
    network.add(layers.Dense(100, activation='relu'))

    # Añadimos una capa softmax para que podamos clasificar las imágenes
    network.add(layers.Dense(5, activation='softmax'))

    network.summary()

    network.compile(optimizer="rmsprop",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return network


def store_network(network):
    print("Guardando modelo")
    m_json = network.to_json()
    with open(globals()['NET_MODEL'], "w") as json_file:
        json_file.write(m_json)
    network.save_weights(globals()['NET_WEIGHTS'])
    print("Modelo Guardado")


# Cargamos la red
def load_network(file_path):
    print("Cargando modelo")
    file = open(file_path, "r")
    network_json = file.read()
    file.close()
    print("Modelo Cargado")
    return model_from_json(network_json)


# Cargamos pesos en el modelo
def load_weights(model, file_path):
    print("Cargando pesos")
    model.load_weights(file_path)
    print("Pesos Cargados")
    return model


def get_saved_network():
    _network = load_network(globals()['NET_MODEL'])
    _network = load_weights(_network, globals()['NET_WEIGHTS'])
    return _network


def predict(network, img):
    prediction = network.predict(img)[0]
    print("Prediction")
    print(prediction)
    return prediction
