#!/usr/bin/python3

import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
from model import runModel

app = Flask(__name__)

@app.route('/')
def home():
    return 'API Modelling'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    return jsonify({'Status':'Data Collected', 'Prediction': runModel(df, typed='single')})

if __name__ == '__main__':
    app.run()