from flask import Flask, render_template, request
import json
import os
import pickle
import numpy as np
import pandas as pd
import joblib
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

best_model = joblib.load('model_predict/best_model.pkl')


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html') 

@app.route('/predict', methods=['POST'])

def home():
    text = request.form['input']
    request_model = request.form['model']

    X = word_tokenize(text, format="text")
    features = tfidf.transform([X]).toarray()
    
    if request_model == "1":
        model = best_model
        
    print(X)
    print(request_model)  
    print(model)
    
    pred = model.predict_proba(features) [:,1]

    return render_template('after.html', proba = pred)

if __name__ == "__main__":
    app.run(debug=False)

