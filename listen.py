import numpy as np
import speech_recognition as sr
import tensorflow as tf
import numpy as np
from flask import Flask, send_from_directory
from flask_cors import CORS                                         
import threading
import os
import logging
from sys import argv
import matplotlib.pyplot as plt
import utils
import constants
import pyaudio
from collections import deque
import io
import wave
from time import sleep

direction = "n/a"

'''
------------------------------------------------------------------------------------------------------------------------------

HTTP SERVER CODE

------------------------------------------------------------------------------------------------------------------------------
'''

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def run_http_server():
    global direction
    app = Flask(__name__, static_folder='./dist')
    CORS(app)

    # return the last inferred direction
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

    # serve the flask application
    app.run(
        host=constants.webserver_host_name, 
        port=constants.webserver_port, 
        debug=True, 
        use_reloader=False
    )


'''
------------------------------------------------------------------------------------------------------------------------------

MACHINE LEARNING CODE

------------------------------------------------------------------------------------------------------------------------------
'''

tf.config.optimizer.set_jit(True)
model_a = model_b = input_details = output_details = None

commands_a = ['eight', 'zero', 'right', 'down', 'left', 'two', '_background_noise_', 'stop', 'go', 'up']
commands_b = ['bed', 'right', 'down', 'left', '_background_noise_', 'no', 'wow', 'up', 'yes', 'five']

# Initialize a Tensorflow Lite model from a provided path
def init_tflite_model(path):
    tflite_model = tf.lite.Interpreter(model_path=path)
    tflite_model.allocate_tensors()
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    return tflite_model, input_details, output_details 

# Take in an audio binary (wav file) and return a spectrogram
def prepare_data(audio_binary):
    waveform = utils.decode_audio(audio_binary)
    spectrogram = utils.get_spectrogram(waveform)

    return spectrogram

# Classify the provided audio data using the provided model and return
# the result if the constraints are met
def run_model(inference_array, model, commands, results, index):
    model.set_tensor(input_details[0]['index'], inference_array)
    model.invoke()

    result = model.get_tensor(output_details[0]['index'])
    prediction = np.argmax(result, axis=1)

    thresh_adjust = 0.0
    if commands[int(prediction[0])] == "up":
        thresh_adjust = 0.25

    if result[0][prediction] > 0.45 + thresh_adjust:
        direction = commands[int(prediction[0])]
        results[index] = direction
    else:
        results[index] = "n/a"

# Take a spectrogram and show it using matplotlib
def show_spectrogram(spectrogram):
    _, axes = plt.subplots(2, figsize=(12, 8))
    utils.plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()

# Infer the intended keyword from a provided audio binary
def infer_from_speech(audio_binary):
    # print('Audio binary received. Starting inference..')
    spectrogram = prepare_data(audio_binary)

    inference_array = []
    inference_array.append(spectrogram.numpy())
    inference_array = np.array(inference_array)

    results = {
        'a': None,
        'b': None
    }

    thread_a = threading.Thread(
        target=run_model,
        args=(inference_array, model_a, commands_a, results, 'a'),
        name="inference_a"
    )

    thread_b = threading.Thread(
        target=run_model,
        args=(inference_array, model_b, commands_b, results, 'b'),
        name="inference_b"
    )

    # We chose a threaded model for the inference step because the
    # total classification time decreased from ~4ms to ~2ms
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    result_a = results['a']
    result_b = results['b']

    # print(f'result_a: {result_a}\nresult_b: {result_b}\n')

    prediction = result_a if result_a == result_b else 'n/a'
    return prediction

def listen(device_index, buffer):
    CHUNK = 500
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        output = False,
        frames_per_buffer = CHUNK,
        input_device_index=device_index
    )

    while True:
        data = stream.read(CHUNK)
        buffer.append(data)
        if (len(buffer) < 32):
            continue
        buffer.popleft()


def infer(buffer):
    global direction
    predictions = deque()

    while True:
        sleep(0.008)
        wav_data = None
        with io.BytesIO() as wav_file:
            wav_writer = wave.open(wav_file, "wb")
            try:  # note that we can't use context manager, since that was only added in Python 3.4
                wav_writer.setframerate(16000)
                wav_writer.setsampwidth(2)
                wav_writer.setnchannels(1)
                wav_writer.writeframes(b"".join(buffer))
                wav_data = wav_file.getvalue()
            finally:  # make sure resources are cleaned up
                wav_writer.close()

        predition = infer_from_speech(wav_data)
        predictions.append(predition)
        if (len(predictions) > 2):
            predictions.popleft()

        if len(predictions) == 2 and predictions[0] == predictions[1] and direction != predictions[0]:
            print(f'predition: {direction}')
            direction = predictions[0]

# Listen and infer keywords from the selected microphone
def run_inference():
    print(sr.Microphone.list_microphone_names())

    device_index = 0

    if (len(argv) > 1):
      device_index = int(argv[1])

    buffer = deque()

    print('Starting http server thread...')
    threading.Thread(
        target=lambda: listen(device_index, buffer),
        name="listen_thread"
    ).start()

    print('Starting http server thread...')
    threading.Thread(
        target=lambda: infer(buffer),
        name="infer_thread"
    ).start()

# main method
if __name__ == '__main__':
    print(f'Tensorflow version: {tf.__version__}')
    print(f'Numpy version: {np.__version__}')

    print('Initializing Tensorflow Lite models...')
    model_a, input_details, output_details  = init_tflite_model('model_a.tflite')
    model_b, _, _ = init_tflite_model('model_b.tflite')

    print('Starting http server thread...')
    threading.Thread(
        target=lambda: run_http_server(),
        name="snake_http_server"
    ).start()

    print('Starting inference thread...')
    threading.Thread(
        target=lambda: run_inference(),
        name="inference_thread"
    ).start()