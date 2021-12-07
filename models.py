import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow._api.v2 import audio

from tensorflow.keras import layers
from tensorflow.keras import models

from utils import get_commands

# https://www.tensorflow.org/tutorials/audio/simple_audio
# First model implemented in the program. Cut and paste from
# the Tensorflow article defined above on speech recognition
def shallow_cnn(spectrogram_ds):
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
# The second model implemented in the program. Taken from the 
# the article defined above with little to no modification
def deep_cnn_slim(spectrogram_ds):
  for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
  print('Input shape:', input_shape)
  num_labels = len(get_commands())

  print(num_labels)

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

# This is the model that provided the best results. It's based
# on the model defined above but with a few extra dense layers
def deep_cnn(spectrogram_ds):
  for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
  print('Input shape:', input_shape)
  num_labels = len(get_commands())

  print(num_labels)

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
      layers.Conv2D(128, (3, 3)),
      layers.Activation('relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(0.5),
      layers.Flatten(),
      layers.Dense(512),
      layers.Activation('relu'),
      layers.Dense(512),
      layers.Activation('relu'),
      layers.Dense(512),
      layers.Activation('relu'),
      layers.Dense(256),
      layers.Activation('relu'),
      layers.Dropout(0.25),
      layers.Dense(num_labels, activation='softmax'),
  ])

  return model

# While this model is closely related to the model defined above it
# started as a derivative of the VGG model. Throughout tuning this
# model, we figured out how similar it was to the model we already
# had so we ended up not really using it. It will stay here for
# posterity's sake.
def deep_cnn_b(spectrogram_ds):
  for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
  print('Input shape:', input_shape)
  num_labels = len(get_commands())

  print(num_labels)

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
      layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(64,64,3), activation="relu"),
      # layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu"),
      layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
      layers.Dropout(0.25),
      layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
      layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
      layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
      layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
      layers.Dropout(0.5),
      layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
      layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
      layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
      layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
      layers.Dropout(0.5),
      layers.Flatten(),
      layers.Dense(units=512,activation="relu"),
      layers.Dropout(0.25),
      layers.Dense(units=512,activation="relu"),
      layers.Dropout(0.25),
      layers.Dense(units=512,activation="relu"),
      layers.Dense(units=512,activation="relu"),
      layers.Dense(num_labels, activation='softmax')

  ])

  return model