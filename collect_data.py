import speech_recognition as sr
from sys import argv

import constants

i = 0

def collect_sample(r, source):
    global i
    print("Say something!")
    audio = r.listen(source, phrase_time_limit=1)
    wav_data = audio.get_wav_data()
    file = open(f"data/{argv[1]}/sample-thomas-{i}.wav", "wb")
    file.write(wav_data)
    file.close()
    print('speech written')
    print(f'count = {i}')
    i += 1

if __name__ == "__main__":
  for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

  r = sr.Recognizer()
  r.energy_threshold = constants.mic_threshold
  print(f"\n\nCOLLECTION DATA FOR: {argv[1]}\n\n")
  with sr.Microphone(device_index=2) as source:
        r.adjust_for_ambient_noise(duration=5, source=source)
        while 1:
            collect_sample(r, source)
            if i == 115:
              print("stopping data collection")
              break