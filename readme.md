# Snake, but dumb

## The Dirty Details
1. We used the Google speech_commands dataset as a starting point for this project so I would recommend you start there as well.

    Dataset download: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

    Store dataset contents inside of a folder named `data` in the root of the project

2. This project was built in an anaconda environment and all packages can be installed using the command `conda env create -f environment.yml` outside of `tensorflow_model_optimization` which was installed using pip.

3. If you want to collect your own data, collect_data.py will help you do just that.

4. There are 3 models to choose from when training: shallow_cnn, deep_cnn, deep_cnn slim. We found the best results with deep_cnn and deep_cnn_slim but your milage may vary. The model used to train can be set in train.py.

5. Use listen.py after training the model to run the game and start listening from your selected audio device. You may need to manually set the audio device you want to listen on, in which case you should run the command as such: `python listen.py <index_of_audio_device>`. If you run listen.py without an explicit index, it will default to 0 but there will also be a printout of all of the devices present somewhere in the output.

6. Listen.py will also spawn a webserver that will host the snake game. This can be found at http://localhost:23336/app.

## How It Works

We used a library named [speechrecognition](https://pypi.org/project/SpeechRecognition/) to do most of the heavy lifting on the mic side. This let us set thresholds to get wav files from the mic in near real time. From there, we take the wav files and convert them into spectrograms and then that goes into the pre-trained tensorflow-lite model. From there, if the resulting classification passes the 40% certainty threshold, the new direction is served to the web application.

## Acknowledgements

[react-simple-snake](https://github.com/MaelDrapier/react-simple-snake)
[Medium: Urban Sound Classification using Convolutional Neural Networks](https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4)
[Tensorflow: Recognizing Keywords](https://www.tensorflow.org/tutorials/audio/simple_audio#build_and_train_the_model)