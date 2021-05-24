import pathlib

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Подготовка данных
datagen_test = ImageDataGenerator(rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

train_it = datagen_test.flow_from_directory('dataset/training_set/training_set', class_mode='binary', batch_size=64)
x, y = train_it.next()
for i in range(0, 1):
    image = x[i]
    plt.imshow(image.astype("uint8"))
    plt.show()
