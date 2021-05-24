import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from keras.constraints import maxnorm
from keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, MaxPool2D

from tensorboard.util import encoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras.optimizer_v1 import adam
from tensorflow.python.keras.regularizers import l2

IMG_WIDTH = 100
IMG_HEIGHT = 100
images = r'N:\Лабы АН\neural-networks-labs\lr1\dataset\fruits\fruits-360\Training'
images_test = r'N:\Лабы АН\neural-networks-labs\lr1\dataset\fruits\fruits-360\Test'


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

fruits = len(class_names_dict.keys())

# создаем список меток
classes = [class_names_dict[class_names[i]] for i in range(len(class_names))]
classes_test = [class_names_dict_test[class_names_test[i]] for i in range(len(class_names_test))]
train_labels = tf.cast(list(map(int, classes)), tf.int32)
test_labels = tf.cast(list(map(int, classes_test)), tf.int32)
# # train_labels = np.asarray(classes).astype('float32').reshape((-1,1))
# # test_labels = np.asarray(classes_test).astype('float32').reshape((-1,1))
#
#  преобразуем всё в nparray
train_images = np.array([np.array(elem) for elem in img_data])
test_images = np.array([np.array(elem) for elem in img_data_test])

l2_reg = 0.001
opt = adam(lr=0.001)
# Defining the CNN Model
cnn_model = Sequential()
cnn_model.add(
    Conv2D(filters=32, kernel_size=(2, 2), input_shape=(100, 100, 3), activation='relu', kernel_regularizer=l2(l2_reg)))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(l2_reg)))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', kernel_regularizer=l2(l2_reg)))
cnn_model.add(MaxPool2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.1))

cnn_model.add(Flatten())

cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(16, activation='relu'))
cnn_model.add(Dense(131, activation='softmax'))

cnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = cnn_model.fit(train_images, test_images, batch_size=128, epochs=110, verbose=1, validation_split=0.33)

scores = cnn_model.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
