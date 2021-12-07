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

def init_tflite_model(path):
    tflite_model = tf.lite.Interpreter(model_path=path)
    tflite_model.allocate_tensors()
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    return tflite_model, input_details, output_details 

# listen to the defined microphone 
def collect_speech(r, source, convert_rate=None):
    print("Say something!")
    audio = r.listen(source, phrase_time_limit=0.75)
    wav_data = audio.get_wav_data(convert_rate)
    return wav_data

def prepare_data(audio_binary):
    waveform = utils.decode_audio(audio_binary)
    spectrogram = utils.get_spectrogram(waveform)

    return spectrogram

def run_model(inference_array, model, commands, results, index):
    model.set_tensor(input_details[0]['index'], inference_array)
    model.invoke()

    result = model.get_tensor(output_details[0]['index'])
    prediction = np.argmax(result, axis=1)
    if result[0][prediction] > 0.4:
        direction = commands[int(prediction[0])]
        results[index] = direction
    else:
        results[index] = "n/a"

def show_spectrogram(spectrogram):
    _, axes = plt.subplots(2, figsize=(12, 8))
    utils.plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()

def infer_from_speech(audio_binary):
    print('Audio binary received. Starting inference..')
    global direction
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

    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    result_a = results['a']
    result_b = results['b']

    print(f'result_a: {result_a}\nresult_b: {result_b}\n')

    direction = result_a if result_a == result_b else 'n/a'

def run_inference():
    # obtain audio from the microphone
    r = sr.Recognizer()
    r.phrase_threshold = 0.125
    r.pause_threshold = 0.175
    r.non_speaking_duration = 0.175

    print(sr.Microphone.list_microphone_names())

    device_index = 0

    if (len(argv) > 1):
      device_index = int(argv[1])

    try:
        with sr.Microphone(device_index, sample_rate=16000) as source:
            r.adjust_for_ambient_noise(duration=4, source=source)
            while 1:
                audio_binary = collect_speech(r, source)
                infer_from_speech(audio_binary)
    except:
        with sr.Microphone(device_index) as source:
            r.adjust_for_ambient_noise(duration=4, source=source)
            while 1:
                audio_binary = collect_speech(r, source, 16000)
                infer_from_speech(audio_binary)

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