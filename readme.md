# Snake, but dumb

## What?

This is the game of snake, but you control it with your voice. We use a 2 tensorflow lite models that contain a derivative of VGG and that have ~500k trainable params each in an ensemble arrangement for the real-time inference. We also use the bagging method to provide each model with a completely different dataset save for the 4 activation phrases we really wanted to focus on. With both models running simultaneously in different threads, we have generally seen an inference time of ~2ms for a 1 second audio binary. We observed ~100ms as being the total time it takes from starting to speak to having a classification of that audio.

## The Dirty Details
1. We used the Google speech_commands dataset as a starting point for this project so I would recommend you start there as well.

    Dataset download: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

    Store dataset contents inside of a folder named `data` in the root of the project

2. This project was built in an anaconda environment and all packages can be installed using the command `conda env create -f environment.yml` outside of `tensorflow_model_optimization` which was installed using pip.

3. If you want to collect your own data, collect_data.py will help you do just that.

4. There are 3 models to choose from when training: shallow_cnn, deep_cnn, deep_cnn slim. We found the best results with deep_cnn and deep_cnn_slim but your milage may vary. The model used to train can be set in train.py.

5. Use listen.py after training the model to run the game and start listening from your selected audio device. You may need to manually set the audio device you want to listen on, in which case you should run the command as such: `python listen.py <index_of_audio_device>`. If you run listen.py without an explicit index, it will default to 0 but there will also be a printout of all of the devices present somewhere in the output.

6. Listen.py will also spawn a webserver that will host the snake game. This can be found at http://localhost:23336/app.

## Roadmap

- [ ] Turn the listening program into a class
- [ ] Attempt a rewrite in Rust to produce a smaller exec

## How It Works

We record audio directly from the selected microphone at 16 kHz and buffer ~1s of audio for inference. When the incoming audio is above a reasonable activity threshold you start the inference engine and run it for 1.5 seconds after the audio level decreases again. While the inference engine is running, we take the entire buffer from the listener and encode it to wave format every 6ms. From there, we take the wav file and convert the audio into a spectrogram and then that goes into the pre-trained Tensorflow lLite models. From there, if the resulting classification passes the 40% certainty threshold, the new direction is served to the web application.

## Acknowledgements

[react-simple-snake](https://github.com/MaelDrapier/react-simple-snake)

[Medium: Urban Sound Classification using Convolutional Neural Networks](https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4)

[Tensorflow: Recognizing Keywords](https://www.tensorflow.org/tutorials/audio/simple_audio#build_and_train_the_model)