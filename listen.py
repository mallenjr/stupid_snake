import speech_recognition as sr

# main method
if __name__ == '__main__':

    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while 1:
            print("Say something!")
            audio = r.listen(source)