import numpy as np
import speech_recognition as sr
import utils
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from flask import Flask
from flask_cors import CORS                                               
import threading
from random import randint

model = load_model('model')

commands = utils.get_commands()
print(commands)

try:
  AUTOTUNE = tf.data.AUTOTUNE     
except:
  AUTOTUNE = tf.data.experimental.AUTOTUNE 

host_name = "0.0.0.0"
port = 23336
app = Flask(__name__)
CORS(app)

direction = "right"

@app.route("/")
def main():
    global direction
    return direction

# main method
if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()

    # obtain audio from the microphone
    r = sr.Recognizer()
    r.energy_threshold = 600
    with sr.Microphone() as source:
        while 1:
            print("Say something!")
            audio = r.listen(source, phrase_time_limit=1)
            wav_data = audio.get_wav_data(convert_rate=11025)
            decoded_audio, _ = tf.audio.decode_wav(contents=wav_data, desired_samples=16000)
            prepared_audio = tf.squeeze(decoded_audio, axis=-1)
            spectrogram = utils.get_spectrogram(prepared_audio)


            file = open("data/out.wav", "wb")
            file.write(wav_data)
            file.close()

            test = ["data/out.wav"]
            output_ds = utils.preprocess_dataset(test)

            test_audio = []
            test_labels = []

            for audio, label in output_ds:
                test_audio.append(audio.numpy())
                test_labels.append(label.numpy())

            test_audio = np.array(test_audio)
            test_labels = np.array(test_labels)

            result = model.predict(test_audio)
            prediction = np.argmax(result, axis=1)
            if result[0][prediction] > 0.4:
                print(prediction[0])
                print(commands[int(prediction[0])])
                direction = commands[int(prediction[0])]