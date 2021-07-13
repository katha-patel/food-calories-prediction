from __future__ import division, print_function
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from tensorflow.keras.models import Sequential

from calorie import calories

from cnn_model import get_model
global graph,fruit_calories
global file_path,model
import tensorflow as tf
graph = tf.compat.v1.get_default_graph
app = Flask(__name__)

IMG_SIZE = 400
LR = 1e-3
no_of_fruits=7

MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, '5conv-basic')

model_save_at=os.path.join("models",MODEL_NAME)

model=get_model(IMG_SIZE,no_of_fruits,LR)

model.load(model_save_at)
#model = load_model(model_save_at)

def model_predict(img_path, model):
    labels = ["Apple", "Banana", "Carrot", "Cucumber", "Onion", "Orange", "Tomato"]
    img = cv2.imread(img_path)
    img1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    model_out = model.predict([img1])
    result = np.argmax(model_out)
    name = labels[result]
    cal = round(calories(result + 1, img), 2)
    return '{} calories:{}'.format(name,cal)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        #result = ls[preds[0]]
        #print(preds)
        return preds
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
