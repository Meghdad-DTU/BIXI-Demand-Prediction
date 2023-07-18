import sys
from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

#sys.path.append("/home/paladin/Downloads/BIXI-Demand-Prediction/")
from src.pipeline.predict_pipline import CustomData, PredictionPipeline


app = Flask(__name__)

# Route for a home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
        data = request.json['data']
        data=CustomData(
            com1=data['com1'],
            com2=data['com2'],
            com3=data['com3'],
            com4=data['com4'],
            com5=data['com5'],
            com6=data['com6']
        )

        df=data.get_data_as_data_frame()
        assert len(df)>96, 'Warning: Number of timstamps must be greater than 96'
        
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(df)
        # If you try to serialize a NumPy array to JSON in Python, you'll get error.
        return jsonify(results.tolist())        

if __name__ == '__main__':
    app.run(debug=True)
    
    

