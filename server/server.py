import base64
import random 
import time
import os
import json
from flask import Flask, request, safe_join, send_file # , Response

app = Flask(__name__, static_folder='models')
app.config['MODEL_PATH'] = 'models/'
app.config['TRAIN_PATH'] = 'train/'


@app.route('/')
def ping():
    return { 'result': 1, 'msg': 'SERVER_IS_RUNNING' }

@app.route('/upload', methods = ['POST'])
def upload():
    #
    # TODO: Add RBAC
    #
    try:
        image_label  = request.form['label']
        image_base64 = request.form['image']
        image_bin = base64.b64decode(image_base64)
        DIR = f'{app.config["TRAIN_PATH"]}/{image_label}'
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
        with open(f'{DIR}/{int(time.time()*1000)}_{int(random.random()*1000)}.jpg', 'wb') as f:
            f.write(image_bin)
        return { 'result': 1 }
    except Exception as e:
        return { 'result': 0, 'err': str(e) }, 400

@app.route('/train', methods = ['POST'])
def go_train():
    # training goes here, but should trigger an async task
    # as user could not wait and http request cannot hang so long
    return { 'result': 1, 'model_name': 'face.model' }

@app.route('/status', methods = ['GET'])
def status():
    return { 'result': 1, 'status': 'training' }

@app.route('/model/<filename>', methods = ['GET'])
def download_model(filename):
    try:
        #
        # WARNING: unsafe access
        #
        safe_path = safe_join(app.config["MODEL_PATH"], filename)
        return send_file(safe_path, as_attachment=True)
    except Exception as e:
        return { 'result': 0, 'err': str(e) }, 400

if __name__ == '__main__':
    app.run(port=8080)