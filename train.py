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

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = './data'

data_dir = pathlib.Path(DATASET_PATH)

def load_single_audio_file():

  f = tf.io.read_file('./data/bed/1aed7c6d_nohash_0.wav')

  audio, _ = tf.audio.decode_wav(contents=f)
  print(audio.shape)

  parts = tf.strings.split(
      input='./data/bed/1aed7c6d_nohash_0.wav',
      sep=os.path.sep)

  print(parts[-2])

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)



def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_audio_files():

  commands = np.array(tf.io.gfile.listdir(str(data_dir)))
  commands = commands[commands != 'README.md']
  print('Commands:', commands)



  filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
  filenames = tf.random.shuffle(filenames)
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


  try:
    AUTOTUNE = tf.data.AUTOTUNE     
  except:
    AUTOTUNE = tf.data.experimental.AUTOTUNE 

  files_ds = tf.data.Dataset.from_tensor_slices(train_files)

  waveform_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)


  rows = 3
  cols = 3
  n = rows * cols
  fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

  for i, (audio, label) in enumerate(waveform_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

  plt.show()




if __name__ == "__main__":
  plot_audio_files()