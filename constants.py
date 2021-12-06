import speech_recognition
import tensorflow as tf

try:
  AUTOTUNE = tf.data.AUTOTUNE     
except:
  AUTOTUNE = tf.data.experimental.AUTOTUNE 

webserver_host_name = "0.0.0.0"
webserver_port = 23336

mic_threshold = 400

batch_size = 64