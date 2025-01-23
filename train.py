import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# load features
try:
    with open('features.pkl', 'rb') as file:
        X_train, X_val, y_train_log, y_val_log = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'features.pkl' was not found. Make sure it is in the correct directory.")

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
model_xgb_final.fit(X_train, y_train_log)

# predictions on validation set
y_pred_val_log = model_xgb_final.predict(X_val)
y_pred_val = np.expm1(y_pred_val_log)
y_val_actual = np.expm1(y_val_log)

# evaluate
rmse = sqrt(mean_squared_error(y_val_actual, y_pred_val))
mse = mean_squared_error(y_val_actual, y_pred_val)
mae = mean_absolute_error(y_val_actual, y_pred_val)
r2 = r2_score(y_val_actual, y_pred_val)

# print
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")


# save final model
try:
    with open('final_xgboost_model.pkl', 'wb') as file:
        pickle.dump(model_xgb_final, file)
    print("Final model trained and saved as 'final_xgboost_model.pkl'")
except Exception as e:
    print(f"Error saving the model: {e}")
