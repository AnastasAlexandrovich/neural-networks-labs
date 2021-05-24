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


IMG_WIDTH = 100
IMG_HEIGHT = 100
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
# train_labels = np.asarray(classes).astype('float32').reshape((-1,1))
# test_labels = np.asarray(classes_test).astype('float32').reshape((-1,1))

# убираем цвет
train_images_bw = np.array([np.array(elem)[:,:,0] for elem in img_data])
test_images_bw = np.array([np.array(elem)[:,:,0] for elem in img_data_test])

# 1 цветовой канал
train_images_bw = train_images_bw.reshape(8005, 100, 100, 1)
test_images_bw = test_images_bw.reshape(2023, 100, 100, 1)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

model.fit(train_images_bw, train_labels, epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(test_images_bw, test_labels)

print('\nTest accuracy:', test_acc)