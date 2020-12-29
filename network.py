from keras import models
from keras import layers

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

