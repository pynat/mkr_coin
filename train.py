import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# load features
try:
    with open('features.pkl', 'rb') as file:
        X_train, X_val, y_train_log, y_val_log, feature_names = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'features.pkl' was not found. Make sure it is in the correct directory.")

if not feature_names or len(feature_names) != X_train.shape[1]:
    raise ValueError("Feature names are missing or do not match the number of columns in X_train.")

# now using whole dataset
X_full = np.vstack([X_train, X_val])  
y_full_log = np.concatenate([y_train_log, y_val_log]) 


# best parameters 
best_xgb_params = {
    'eta':  0.25,
    'max_depth': 5,
    'min_child_weight': 1, 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}


# train final model
model_xgb_final = xgb.XGBRegressor(**best_xgb_params)
model_xgb_final.fit(X_full, y_full_log)

# assign feature names explicitly
model_xgb_final.get_booster().feature_names = feature_names

# save final model
try:
    with open('final_xgboost_model.pkl', 'wb') as file:
        pickle.dump(model_xgb_final, file)
    print("Final model trained and saved as 'final_xgboost_model.pkl'")
except Exception as e:
    print(f"Error saving the model: {e}")
