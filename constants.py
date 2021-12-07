from tensorflow import data

try:
  AUTOTUNE = data.AUTOTUNE     
except:
  AUTOTUNE = data.experimental.AUTOTUNE 

webserver_host_name = "0.0.0.0"
webserver_port = 23336

mic_threshold = 600

batch_size = 64

DATASET_PATH = './data_b'
