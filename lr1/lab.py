import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
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
class_names_dict_test = {k: v for v, k in enumerate(np.unique(class_names_test))}
# создаем список меток
classes = [class_names_dict[class_names[i]] for i in range(len(class_names))]
classes_test = [class_names_dict_test[class_names_test[i]] for i in range(len(class_names_test))]

# отобразим первые 25 изображений из обучающего набора и отобразим имя класса под каждым изображением.
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(img_data[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[i])
# plt.show()

# создаем нашу модель
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# компилируем нашу модель
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# обучаем модель
images_data_in_numbers = tf.cast(np.array(img_data), tf.float32)
images_data_in_numbers_test = tf.cast(np.array(img_data_test), tf.float32)
# images_data_in_numbers = tf.transpose(images_data_in_numbers, perm=[0, 1, 2, 3])
labels = tf.cast(list(map(int, classes)), tf.int32)
labels_test = tf.cast(list(map(int, classes_test)), tf.int32)
print(labels_test)
model.fit(images_data_in_numbers, labels, epochs=10)

# оцениваем

test_loss, test_acc = model.evaluate(images_data_in_numbers_test, labels_test, verbose=2)

print('\nTest accuracy:', test_acc)







