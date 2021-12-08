from flask import Flask, send_from_directory
from flask_cors import CORS                                         
import threading
import logging
from sys import argv
import sys
import utils
import constants
from collections import deque
from io import BytesIO
from wave import open as wave_open
from pyaudio import PyAudio
from math import ceil, floor
from time import sleep, perf_counter
from tensorflow import lite
from os import path as os_path
from numpy import argmax, int16, frombuffer, ndarray
from numpy import array as np_array
import webbrowser
from itertools import islice

direction = "n/a"

'''
------------------------------------------------------------------------------------------------------------------------------

HTTP SERVER CODE

------------------------------------------------------------------------------------------------------------------------------
'''


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os_path.abspath(".")

    return os_path.join(base_path, relative_path)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def run_http_server():
    global direction
    app = Flask(__name__, static_folder=resource_path('./dist'))
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
        if path != "" and os_path.exists(app.static_folder + '/' + path):
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

model_a = model_b = input_details = output_details = None

commands_a = ['eight', 'zero', 'right', 'down', 'left', 'two', '_background_noise_', 'stop', 'go', 'up']
commands_b = ['bed', 'right', 'down', 'left', '_background_noise_', 'no', 'wow', 'up', 'yes', 'five']

# Initialize a Tensorflow Lite model from a provided path
def init_tflite_model(path):
    tflite_model = lite.Interpreter(model_path=path)
    tflite_model.allocate_tensors()
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    return tflite_model, input_details, output_details 

# Classify the provided audio data using the provided model and return
# the result if the constraints are met
def run_model(inference_array, model, commands, results, index):
    model.set_tensor(input_details[0]['index'], inference_array)
    model.invoke()

    result = model.get_tensor(output_details[0]['index'])
    prediction = argmax(result, axis=1)

    thresh_adjust = 0.0
    if commands[int(prediction[0])] == "up":
        thresh_adjust = 0.45

    if result[0][prediction] > (0.45 + thresh_adjust):
        direction = commands[int(prediction[0])]
        results[index] = direction
    else:
        results[index] = "n/a"

# Infer the intended keyword from a provided audio binary
def execute_models(audio_binary):
    # print('Audio binary received. Starting inference..')
    spectrogram = utils.binary_to_spec(audio_binary)

    inference_array = []
    inference_array.append(spectrogram.numpy())
    inference_array = np_array(inference_array)

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

    prediction = result_a if result_a == result_b else 'n/a'
    return prediction

# Stream audio data from the selected microphone into a 1 second buffer
def listen(device_index, buffer, shared):
    p = PyAudio()

    stream = p.open(
        format = constants.FORMAT,
        channels = constants.CHANNELS,
        rate = constants.RATE,
        input = True,
        output = False,
        frames_per_buffer = constants.CHUNK,
        input_device_index=device_index
    )

    predict_start = perf_counter()

    while True:
        current_time = perf_counter()
        data = stream.read(constants.CHUNK, exception_on_overflow = False)
        buffer.append(data)
        
        if (len(buffer) < ceil(constants.RATE / constants.CHUNK)):
            continue

        data = frombuffer(data, dtype=int16)
        max = ndarray.max(data)
        if (max > 600): # check if current chunk surpasses energy threshold
            predict_start = perf_counter()
            shared['predict'] = True
        if (current_time - predict_start > 1.5): # classify for 1.5 seconds
            shared['predict'] = False

        buffer.popleft()


def predict(buffer, shared):
    global direction
    predictions = deque()

    while True:
        sleep(0.003)
        if (shared['predict'] == False):
            continue
        sleep(0.003)
        wav_data = None
        with BytesIO() as wav_file:
            wav_writer = wave_open(wav_file, "wb")
            try:
                wav_writer.setframerate(constants.RATE)
                wav_writer.setsampwidth(2)
                wav_writer.setnchannels(1)
                wav_writer.writeframes(b"".join(islice(buffer, 0, floor((constants.RATE / constants.CHUNK) / 0.6))))
                wav_data = wav_file.getvalue()
            finally:  # make sure resources are cleaned up
                wav_writer.close()

        predition = execute_models(wav_data)
        predictions.append(predition)
        if (len(predictions) > 2):
            predictions.popleft()

        # Only update direction if the previous two preditions are the same
        if len(predictions) == 2 and predictions[0] == predictions[1] and direction != predictions[0]:
            direction = predictions[0]
            if predictions[0] != 'n/a': print(f'predition: {direction}')


# Listen and infer keywords from the selected microphone
def run_inference():
    p = PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    device_index = 0

    if (len(argv) > 1):
      device_index = int(argv[1])
    else:
      print('\nEnter the device id you would like to use:')
      device_index = int(input())

    buffer = deque()
    shared = {
        'predict': False
    }

    print('Starting listening thread...')
    threading.Thread(
        target=lambda: listen(device_index, buffer, shared),
        name="listen_thread"
    ).start()

    print('Starting predition thread...')
    threading.Thread(
        target=lambda: predict(buffer, shared),
        name="predict_thread"
    ).start()


'''
------------------------------------------------------------------------------------------------------------------------------

EXECUTION CODE

------------------------------------------------------------------------------------------------------------------------------
'''

# main method
if __name__ == '__main__':
    print('Initializing Tensorflow Lite models...')
    model_a, input_details, output_details  = init_tflite_model(resource_path('model_a.tflite'))
    model_b, _, _ = init_tflite_model(resource_path('model_b.tflite'))

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

    sleep(1)
    webbrowser.open_new("http://localhost:23336/app")