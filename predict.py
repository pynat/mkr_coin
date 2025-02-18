import pickle
import numpy as np
from flask import Flask, jsonify, request
import xgboost as xgb

# load the model and components
try:
    with open('final_xgboost_model.pkl', 'rb') as file:
        model_xgb_final = pickle.load(file)
    with open('model_components.pkl', 'rb') as file:
        components = pickle.load(file)
    scaler = components['scaler']
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required model files not found: {e}")

# get feature names
try:
    feature_names = model_xgb_final.get_booster().feature_names
except AttributeError:
    raise ValueError("Feature names could not be loaded. Ensure the model was trained with named features.")

# initialize Flask app
app = Flask(__name__)

# prediction 
def predict_single(features, model, scaler):
    # reshape for a single prediction
    X = np.array(features).reshape(1, -1)
    
    # scale the features
    X_scaled = scaler.transform(X)
    
    # predict
    predicted_close = model.predict(X_scaled)
    
    return predicted_close[0]

# route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # parse JSON request
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        # prepare feature vector
        features = [0] * len(feature_names)
        for key, value in input_data.items():
            if key in feature_names:
                features[feature_names.index(key)] = value

        # predict using model
        prediction = predict_single(features, model_xgb_final, scaler)

        # return result
        result = {
            'predicted_close_price': float(prediction) 
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)