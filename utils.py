from tensorflow import (
  strings,
  squeeze,
  zeros,
  shape,
  concat,
  float32,
  cast,
  newaxis,
  argmax,
)
from tensorflow._api.v2.data import Dataset
from tensorflow._api.v2.signal import stft
from tensorflow._api.v2.io import read_file
from tensorflow._api.v2.io.gfile import listdir
from tensorflow._api.v2.audio import decode_wav
from numpy import squeeze as np_squeeze
from numpy import log as np_log
from numpy import finfo, linspace, size
from numpy import array as np_array
from os import path
import constants
import pathlib

# Get a list of commands for the provided dataset
def get_commands():
  commands = np_array(listdir(str(get_data_dir())))
  commands = commands[commands != 'README.md']
  commands = commands[commands != 'testing_list.txt']
  commands = commands[commands != 'validation_list.txt']
  commands = commands[commands != 'LICENSE']
  commands = commands[commands != 'out.wav']
  return commands

# Return the path for the data directory
def get_data_dir():
  data_dir = pathlib.Path(constants.DATASET_PATH)
  return data_dir

def get_label(file_path):
  parts = strings.split(
      input=file_path,
      sep=path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

# Return a waveform from a provided audio binary
def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return squeeze(audio, axis=-1)


# Return the label and waveform from a provided
# wavfile path
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

# Generate a spectrogram from a supplied waveform
def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = zeros(
      [16000] - shape(waveform),
      dtype=float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = cast(waveform, dtype=float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., newaxis]
  return spectrogram

# Plot a provided spectrogram using matplotlib
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np_squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np_log(spectrogram.T + finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = linspace(0, size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# Return the label and spectrogram from a provided
# waveform and label
def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = argmax(cast(label == get_commands(), float32))
  return spectrogram, label_id

# Processing function used by the training model
def preprocess_dataset(files):
  files_ds = Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=constants.AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=constants.AUTOTUNE)
  return output_ds

# Take in an audio binary (wav file) and return a spectrogram
def binary_to_spec(audio_binary):
    waveform = decode_audio(audio_binary)
    spectrogram = get_spectrogram(waveform)

    return spectrogram