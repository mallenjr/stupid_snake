import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import io, keras

import utils
import constants
import models as m
import tensorflow_model_optimization as tfmot

# Tensorflow setup code. We set the seed value for experiment 
# reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Set a few application constants
data_dir = pathlib.Path(constants.DATASET_PATH)
commands = utils.get_commands()

# Get the list of filenames to be used for
# training/validation and return it
def get_filenames():
  filenames = io.gfile.glob(str(data_dir) + '/*/*')
  filenames = tf.random.shuffle(filenames)

  print(len(filenames))

  print(commands)

  num_samples = len(filenames)
  print('Number of total examples:', num_samples)
  print('Number of examples per label:',
        len(io.gfile.listdir(str(data_dir/commands[0]))))
  print('Example file tensor:', filenames[0])

  return filenames

# Prepare the various dataset used for training/validation
# and return the individual datasets as well as the lambda
# function for use in the normalization layer of the training
# model
def prepare_datasets(filenames):
  size_a = int(len(filenames) * 0.8)
  size_b = int(len(filenames) * 0.1)

  train_files = filenames[:size_a]
  val_files = filenames[size_a: size_a + size_b]
  test_files = filenames[size_a + size_b:]

  print('Training set size', len(train_files))
  print('Validation set size', len(val_files))
  print('Test set size', len(test_files))

  files_ds = tf.data.Dataset.from_tensor_slices(train_files)

  # Take a list of files and convert the entire dataset to
  # spectrograms
  waveform_ds = files_ds.map(
      map_func=utils.get_waveform_and_label,
      num_parallel_calls=constants.AUTOTUNE)

  spectrogram_ds = waveform_ds.map(
    map_func=utils.get_spectrogram_and_label_id,
    num_parallel_calls=constants.AUTOTUNE)

  train_ds = spectrogram_ds
  val_ds = utils.preprocess_dataset(val_files)
  test_ds = utils.preprocess_dataset(test_files)

  batch_size = constants.batch_size
  train_ds = train_ds.batch(batch_size)
  val_ds = val_ds.batch(batch_size)

  train_ds = train_ds.cache().prefetch(constants.AUTOTUNE)
  val_ds = val_ds.cache().prefetch(constants.AUTOTUNE)

  return spectrogram_ds, train_ds, val_ds, test_ds

# Train the model and save the Tensorflow Lite model
# to a file to be used in the inference step
def train_model(spectrogram_ds, train_ds, val_ds, test_ds):
  print(tf.__version__)

  model = m.deep_cnn_slim(spectrogram_ds)

  model.summary()

  cluster_weights = tfmot.clustering.keras.cluster_weights
  CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

  clustering_params = {
    'number_of_clusters': 32,
    'cluster_centroids_init': CentroidInitialization.LINEAR
  }

  # Cluster a whole model
  clustered_model = cluster_weights(model, **clustering_params)


  clustered_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
  )

  EPOCHS = 30
  history = clustered_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=keras.callbacks.EarlyStopping(verbose=1, patience=3),
  )

  final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

  converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  tflite_quant_model = converter.convert()

  with open('model_b.tflite', 'wb') as f:
    f.write(tflite_quant_model)
    
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

  y_pred = np.argmax(clustered_model.predict(test_audio), axis=1)
  y_true = test_labels

  test_acc = sum(y_pred == y_true) / len(y_true)
  print(f'Test set accuracy: {test_acc:.0%}')


if __name__ == "__main__":
  filenames = get_filenames()
  spectrogram_ds, train_ds, val_ds, test_ds = prepare_datasets(filenames)
  train_model(spectrogram_ds, train_ds, val_ds, test_ds)