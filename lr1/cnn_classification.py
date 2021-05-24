import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from keras.constraints import maxnorm
from keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D
from tensorboard.util import encoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


IMG_WIDTH = 200
IMG_HEIGHT = 200
images = r'N:\Лабы АН\neural-networks-labs\lr1\dataset\training_set\training_set'
images_test = r'N:\Лабы АН\neural-networks-labs\lr1\dataset\test_set\test_set'


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir)):
            image_path = os.path.join(img_folder, dir, file)
            image = plt.imread(image_path)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255.0
            img_data_array.append(image)
            class_name.append(dir)
    return img_data_array, class_name  # extract the image array and class name


# массив изображений и наимеование каждого класса изображений
img_data, class_names = create_dataset(images)
img_data_test, class_names_test = create_dataset(images_test)

# создаем словарь имя класса - метка (имя - номер)
class_names_dict = {k: v for v, k in enumerate(np.unique(class_names))}
class_names_dict_test = {k: v for v, k in enumerate(np.unique(class_names))}

# создаем список меток
classes = [class_names_dict[class_names[i]] for i in range(len(class_names))]
classes_test = [class_names_dict_test[class_names_test[i]] for i in range(len(class_names_test))]
train_labels = tf.cast(list(map(int, classes)), tf.int32)
test_labels = tf.cast(list(map(int, classes_test)), tf.int32)

#  преобразуем всё в nparray
train_images = np.array([np.array(elem) for elem in img_data])
test_images = np.array([np.array(elem) for elem in img_data_test])

model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function
model.add(keras.layers.Dense(512,activation='relu'))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog
# Here we use sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(train_images, train_labels, epochs=5, batch_size=32)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)
