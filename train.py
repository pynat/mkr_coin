import pickle
import numpy as np
import xgboost as xgb

# load model and components
try:
    with open('model_components.pkl', 'rb') as file:
        components = pickle.load(file)
    
    with open('model_xgb.pkl', 'rb') as file:
        model_xgb = pickle.load(file)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required model files not found: {e}")

# extract components
X_train_scaled = components['X_train_scaled']
X_val_scaled = components['X_val_scaled']
y_train = components['y_train']
y_val = components['y_val']
feature_names = components['feature_names']
scaler = components['scaler']
best_params = components['best_params']

# combine datasets 
X_full_scaled = np.vstack([X_train_scaled, X_val_scaled])
y_full = np.concatenate([y_train, y_val])

# train final model with saved parameters
model_xgb_final = xgb.XGBRegressor(**best_params)
model_xgb_final.fit(X_full_scaled, y_full)

# assign feature names
model_xgb_final.get_booster().feature_names = feature_names

# save final model
try:
    with open('final_xgboost_model.pkl', 'wb') as file:
        pickle.dump(model_xgb_final, file)
    print("Final model trained and saved as 'final_xgboost_model.pkl'")
except Exception as e:
    print(f"Error saving the model: {e}")