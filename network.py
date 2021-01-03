import cv2
from keras import models
from keras import layers
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

import numpy as np

globals()['NET_MODEL'] = "network_model.json"
globals()['NET_WEIGHTS'] = "network_weights.h5"

def build_network():
    model = models.Sequential()

    # Capa entrada
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(516, activation='relu'))
    model.add(Dense(225, activation='relu'))
    model.add(Dense(175, activation='relu'))
    model.add(Dropout(0.25))


    # Añadimos una capa softmax para que podamos clasificar las imágenes
    model.add(layers.Dense(5, activation='softmax'))

    model.summary()

    model.compile(optimizer="rmsprop",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return model


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
    # print("Prediction")
    # print(prediction)
    return prediction
