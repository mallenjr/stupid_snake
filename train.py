import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow._api.v2 import audio

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import utils
import models as m

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = './data'

data_dir = pathlib.Path(DATASET_PATH)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

commands = utils.get_commands()

try:
  AUTOTUNE = tf.data.AUTOTUNE     
except:
  AUTOTUNE = tf.data.experimental.AUTOTUNE 

def load_single_audio_file():

  f = tf.io.read_file('./data/bed/1aed7c6d_nohash_0.wav')

  audio, _ = tf.audio.decode_wav(contents=f)
  print(audio.shape)

  parts = tf.strings.split(
      input='./data/bed/1aed7c6d_nohash_0.wav',
      sep=os.path.sep)

  print(parts[-2])

def plot_audio_files():
  print('Commands:', commands)

  filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
  filenames = tf.random.shuffle(filenames)

  train_files = filenames[:6400]
  val_files = filenames[6400: 6400 + 800]
  test_files = filenames[-800:]


  num_samples = len(filenames)
  print('Number of total examples:', num_samples)
  print('Number of examples per label:',
        len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
  print('Example file tensor:', filenames[0])

  train_files = filenames[:6400]
  val_files = filenames[6400: 6400 + 800]
  test_files = filenames[-800:]

  print('Training set size', len(train_files))
  print('Validation set size', len(val_files))
  print('Test set size', len(test_files))

  files_ds = tf.data.Dataset.from_tensor_slices(train_files)

  waveform_ds = files_ds.map(
      map_func=utils.get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)

  spectrogram_ds = waveform_ds.map(
    map_func=utils.get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)

  train_ds = spectrogram_ds
  val_ds = utils.preprocess_dataset(val_files)
  test_ds = utils.preprocess_dataset(test_files)

  batch_size = 64
  train_ds = train_ds.batch(batch_size)
  val_ds = val_ds.batch(batch_size)

  train_ds = train_ds.cache().prefetch(AUTOTUNE)
  val_ds = val_ds.cache().prefetch(AUTOTUNE)

  for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
  print('Input shape:', input_shape)
  num_labels = len(commands)

  print(tf.__version__)

  model = m.model_a()

  model.summary()

  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
  )

  EPOCHS = 15
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
  )

  metrics = history.history
  plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
  plt.legend(['loss', 'val_loss'])
  plt.show()

  test_audio = []
  test_labels = []

  for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

  test_audio = np.array(test_audio)
  test_labels = np.array(test_labels)

  y_pred = np.argmax(model.predict(test_audio), axis=1)
  y_true = test_labels

  test_acc = sum(y_pred == y_true) / len(y_true)
  print(f'Test set accuracy: {test_acc:.0%}')


if __name__ == "__main__":
  plot_audio_files()