import pickle
import numpy as np
from flask import Flask, jsonify, request
import xgboost as xgb

# load the model
try:
    with open('final_xgboost_model.pkl', 'rb') as file:
        model_xgb_final = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The model file 'final_xgboost_model.pkl' was not found. Ensure it is in the correct directory.")

# dynamically load feature names
try:
    feature_names = model_xgb_final.get_booster().feature_names
except AttributeError:
    raise ValueError("Feature names could not be loaded. Ensure the model was trained with named features.")

# initialize Flask app
app = Flask(__name__)

# prediction function
def predict_single(features, model):
    # reshaping for a single prediction
    X = np.array(features).reshape(1, -1)  
    y_pred_log = model.predict(X)          
    y_pred = np.expm1(y_pred_log)          
    return y_pred[0]

# route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        # Prepare feature vector
        features = [0] * len(feature_names)
        for key, value in input_data.items():
            if key in feature_names:
                features[feature_names.index(key)] = value

        # Predict using the model
        prediction = predict_single(features, model_xgb_final)

        # Return result
        result = {
            'predicted_growth_rate': float(prediction)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
