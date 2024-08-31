from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
# Assume the model is stored as 'thyroid_model.pkl'
file_path = r'path\to\Thyroid_model.pkl'
with open(file_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('home.html')

# Define the route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Process input data as needed
    features = [
        int(data['age']),
        1 if data['sex'] == 'Male' else 0,
        1 if 'on_thyroxine' in data else 0,
        1 if 'query_on_thyroxine' in data else 0,
        1 if 'on_antithyroid_medication' in data else 0,
        1 if 'sick' in data else 0,
        1 if 'pregnant' in data else 0,
        1 if 'thyroid_surgery' in data else 0,
        1 if 'I131_treatment' in data else 0,
        1 if 'query_hypothyroid' in data else 0,
        1 if 'query_hyperthyroid' in data else 0,
        1 if 'lithium' in data else 0,
        1 if 'goitre' in data else 0,
        1 if 'tumor' in data else 0,
        1 if 'hypopituitary' in data else 0,
        1 if 'psych' in data else 0,
        float(data['TSH']),
        float(data['T3']),
        float(data['TT4']),
        float(data['T4U']),
        float(data['FTI']),
        1 if data['referral_source'] == 'SVHC' else 
        2 if data['referral_source'] == 'STMW' else 
        3 if data['referral_source'] == 'SVI' else 4
    ]

    # Convert the features to a numpy array and reshape for the model
    features_array = np.array(features).reshape(1, -1)
    
    # Make the prediction using the loaded model
    prediction = model.predict(features_array)[0]

    return jsonify(result=prediction)

if __name__ == '__main__':
    app.run(debug=True)