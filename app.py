from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
clf = pickle.load(open('hdmodel.pkl','rb'))

@app.route('/', methods = ['GET'])
@cross_origin()
def hello():
    a="it's working"
    return jsonify(a)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictions():
    req_data = request.json

    age=req_data['age']
    sex=req_data['sex']
    cp=req_data['cp']
    trestbps=req_data['trestbps']
    chol=req_data['chol']
    fbs=req_data['fbs']
    restecg=req_data['restecg']
    thalach=req_data['thalach']
    exang=req_data['exang']
    oldpeak=req_data['oldpeak']
    slope=req_data['slope']
    ca=req_data['ca']
    thal=req_data['thal']

    y=np.asarray(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]),dtype='float64')

    predicted = clf.predict([y])
    return jsonify(str(predicted[0]))

if __name__ == '__main__':
    app.run(port=5050,debug=True)