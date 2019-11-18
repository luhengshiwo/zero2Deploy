#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify

# from flask_cors import CORS

app = Flask(__name__)


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/cls/', methods=['POST'])
def sentiment_cls():
    sentence = request.get_data()
    file_vocab = open('data/vocab.txt','r')
    vocab = []
    for line in file_vocab:
        vocab.append(line.strip())
    file_vocab.close()
    words = list(sentence.decode('utf-8').strip())
    ids=[]
    for word in words:
        if word in vocab:
            ids.append(vocab.index(word))
        else:
            ids.append(len(vocab)-1)
    X_new = [ids]
    input_data_json = json.dumps({
        "signature_name": "serving_default",
        "instances": X_new,
    })
    SERVER_URL = 'http://144.202.100.179:8501/v1/models/my_cls_model:predict'
    response = requests.post(SERVER_URL, data=input_data_json)
    response = json.loads(response.text)
    predictions = response['predictions'][0]
    return str(predictions)

if __name__ == '__main__':
    app.run('0.0.0.0',port=5000)