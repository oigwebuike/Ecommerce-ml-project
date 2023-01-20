import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_template 


app = Flask(__name__)

# load model to be used
model = pickle.load(open("./models/skl_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict():
    
    data = request.json['data']
    print(data)
    reshaped = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(reshaped)
    op = model.predict(new_data) # output
    print(model.get_params())
    print(op[0])
    
    
    return jsonify(op[0])

if __name__ == '__main__':
    app.run(debug=True)