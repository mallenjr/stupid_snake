from flask import Flask
from flask_cors import CORS                                               
import threading
from random import randint

host_name = "0.0.0.0"
port = 23336
app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
    i = randint(0,3)
    data = ['up','down','left','right']
    data = data[i]
    return data

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()