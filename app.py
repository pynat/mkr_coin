import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('final_xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'final_xgboost_model.pkl' is in the directory.")
        return None

# Load feature names
@st.cache_data 
def load_feature_names():
    with open('features.pkl', 'rb') as file:
        X_train, X_val, y_train_log, y_val_log, feature_names = pickle.load(file)
    return feature_names

feature_names = load_feature_names()

# Load all images from the folder
@st.cache_data
def load_images_from_folder(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):  
        if file.endswith((".png", ".jpg", ".jpeg")):
            images.append(os.path.join(folder_path, file))
    return images

# Define the most important features (from README)
important_features = [
    "30d_ma", "7d_ma", "7d_volatility", "adx", "rsi", "macd", "ppo",
    "atr", "growth_24h", "growth_72h", "volume_change", "price_change"
]

# Make prediction
def predict(features, model):
    X = np.array(features).reshape(1, -1)  # Reshape input for prediction
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)  # Reverse log transformation
    return y_pred[0]

# Streamlit UI
st.title("XGBoost Growth Rate Predictor")

# ui
tab1, tab2 = st.tabs(["📈 Prediction", "📊 Visualizations"])


# --- TAB 1: Prediction ---
with tab1:
    model = load_model()

    if model:
        user_input = {}

        st.write("### Enter Feature Values")
        for feature in important_features:
            user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.01)

        if st.button("Predict"):
            features = [user_input[f] for f in important_features]
            prediction = predict(features, model)
            st.success(f"Predicted Growth Rate: {prediction:.4f}")


# --- TAB 2: Visualizations ---
with tab2:
    st.write("### Model Feature Distributions & Trends")

    folder_path = "images/" 
    image_files = load_images_from_folder(folder_path)

    if image_files:
        for img_path in image_files:
            st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.warning("No images found in the folder.")