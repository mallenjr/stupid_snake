import numpy as np
import speech_recognition as sr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, send_from_directory
from flask_cors import CORS                                         
import threading
import os

import utils
import constants

model = load_model('model')

commands = ['eight', 'zero', 'right', 'down', 'left', 'two', '_background_noise_', 'stop', 'go', 'up']
print(commands)

direction = "right"

def run_http_server():
    global direction
    app = Flask(__name__, static_folder='./dist')
    CORS(app)

    @app.route("/")
    def main():
        global direction
        return direction

    # Serve React App
    @app.route('/app', defaults={'path': ''})
    @app.route('/app/<path:path>')
    def serve(path):
        print(path)
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    app.run(
        host=constants.webserver_host_name, 
        port=constants.webserver_port, 
        debug=True, 
        use_reloader=False
    )


def collect_speech(r, source):
    print("Say something!")
    audio = r.listen(source, phrase_time_limit=1)
    wav_data = audio.get_wav_data()
    return wav_data

def prepare_data(audio_binary):
    waveform = utils.decode_audio(audio_binary)
    spectrogram = utils.get_spectrogram(waveform)

    return spectrogram

def infer_from_speech(audio_binary):
    print('starting inference')
    global direction
    spectrogram = prepare_data(audio_binary)

    infrence_array = []
    infrence_array.append(spectrogram.numpy())
    infrence_array = np.array(infrence_array)

    result = model.predict(infrence_array)
    prediction = np.argmax(result, axis=1)
    if result[0][prediction] > 0.4:
        print(prediction[0])
        print(commands[int(prediction[0])])
        direction = commands[int(prediction[0])]
    else:
        print('no result found')


def run_infrence():
    # obtain audio from the microphone
    r = sr.Recognizer()
    r.energy_threshold = constants.mic_threshold
    r.phrase_threshold = 0.2

    print(sr.Microphone.list_microphone_names())

    with sr.Microphone(device_index=0) as source:
        r.adjust_for_ambient_noise(duration=4, source=source)
        while 1:
            audio_binary = collect_speech(r, source)
            infer_from_speech(audio_binary)

# main method
if __name__ == '__main__':
    print(f'tensorflow version: {tf.__version__}')
    print(f'numpy version: {np.__version__}')

    threading.Thread(
        target=lambda: run_http_server()
    ).start()

    threading.Thread(
        target=lambda: run_infrence()
    ).start()