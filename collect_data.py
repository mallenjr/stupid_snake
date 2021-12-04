import speech_recognition as sr

import constants

i = 0
command = "left"

def collect_sample(r, source):
    global i
    print("Say something!")
    audio = r.listen(source, phrase_time_limit=1)
    wav_data = audio.get_wav_data()
    file = open(f"data/{command}/sample-{i}.wav", "wb")
    file.write(wav_data)
    file.close()
    print('speech written')
    print(f'count = {i}')
    i += 1

if __name__ == "__main__":
  r = sr.Recognizer()
  r.energy_threshold = constants.mic_threshold
  with sr.Microphone() as source:
        r.adjust_for_ambient_noise(duration=10, source=source)
        while 1:
            collect_sample(r, source)