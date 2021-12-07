import speech_recognition as sr
from sys import argv

import constants

'''
------------------------------------------------------------------------------------------------------------------------------

DATA COLLECTION CODE

------------------------------------------------------------------------------------------------------------------------------
'''

# The starting index of the data collection script.
# This value is included in each sample stored after
# recording
i = 0

# Listen for a phrase and then store it
def collect_sample(r, source):
    global i
    print("Say something!")
    audio = r.listen(source, phrase_time_limit=1)
    wav_data = audio.get_wav_data()
    file = open(f"data/{argv[1]}/sample-{argv[2]}-{i}.wav", "wb")
    file.write(wav_data)
    file.close()
    print('speech written')
    print(f'count = {i}')
    i += 1

# Main function. Parse command args and start a data collection
# run. This function will record 115 samples and then terminate
def main():
  if (len(argv) < 2):
    print('usage: collect_data.py <command> <sample_collection>')
    return

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

if __name__ == "__main__":
  main()