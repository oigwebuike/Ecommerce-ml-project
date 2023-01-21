import pickle
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor as XGBLR
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SKLR

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, app, jsonify, url_for, render_template 


app = Flask(__name__)

# load model to be used
model = pickle.load(open("./models/skl_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    
    data = request.json['data']
    print(data)
    reshaped = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(reshaped)
    op = model.predict(new_data) # output
    print(model.get_params())
    print(op[0])
    
    
    return jsonify(op[0])


@app.route('/predict', methods=['POST'])
def predict():
    
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    
    op = model.predict(final_input)[0] # output
    return render_template("home.html", prediction_output=f"The new estimated yearly spending is: {op}")

if __name__ == '__main__':
    app.run(debug=True)