import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow._api.v2 import audio

from tensorflow.keras import layers
from tensorflow.keras import models

from utils import get_commands

def model_a(spectrogram_ds):
  for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
  print('Input shape:', input_shape)
  num_labels = len(get_commands())

  # Instantiate the `tf.keras.layers.Normalization` layer.
  norm_layer = tf.keras.layers.experimental.preprocessing.Normalization(
    axis=None)
  # Fit the state of the layer to the spectrograms
  # with `Normalization.adapt`.
  norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

  model = models.Sequential([
      layers.Input(shape=input_shape),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      norm_layer,
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(num_labels),
  ])

  return model

# https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4
def model_b(spectrogram_ds):
  for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
  print('Input shape:', input_shape)
  num_labels = len(get_commands())

  # Instantiate the `tf.keras.layers.Normalization` layer.
  norm_layer = tf.keras.layers.experimental.preprocessing.Normalization(
    axis=None)
  # Fit the state of the layer to the spectrograms
  # with `Normalization.adapt`.
  norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

  model = models.Sequential([
      layers.Input(shape=input_shape),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      norm_layer,
      layers.Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)),
      layers.Activation('relu'),
      layers.Conv2D(64, (3, 3)),
      layers.Activation('relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(0.25),
      layers.Conv2D(64, (3, 3), padding='same'),
      layers.Activation('relu'),
      layers.Conv2D(64, (3, 3)),
      layers.Activation('relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(0.5),
      layers.Conv2D(128, (3, 3), padding='same'),
      layers.Activation('relu'),
      layers.Conv2D(128, (3, 3)),
      layers.Activation('relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(0.5),
      layers.Flatten(),
      layers.Dense(512),
      layers.Activation('relu'),
      layers.Dropout(0.5),
      layers.Dense(num_labels, activation='softmax'),
  ])

  return model