import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="🌿 Fertilizer Recommender", layout="centered")
st.title("🌿 Fertilizer Recommendation System")

# Load CSV and preprocess
@st.cache_data
def load_data():
    df = pd.read_csv("fertilizer_data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Train and return model + feature names
@st.cache_resource
def train_model(df):
    df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
    X = df_encoded.drop("fertilizer_name", axis=1)
    y = df_encoded["fertilizer_name"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, X.columns.tolist()

# UI inputs
N = st.number_input("Nitrogen (N)", 0, 200, 90)
P = st.number_input("Phosphorus (P)", 0, 200, 40)
K = st.number_input("Potassium (K)", 0, 200, 41)
temperature = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 70)
moisture = st.slider("Soil Moisture (%)", 0, 100, 30)

soil_options = ["Loamy", "Clay", "Sandy", "Black", "Red", "Alluvial"]
crop_options = [
    "Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
    "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"
]

soil = st.selectbox("Soil Type", soil_options)
crop = st.selectbox("Crop Type", crop_options)

# Load + Train
df = load_data()
model, feature_names = train_model(df)

if st.button("🔍 Recommend Fertilizer"):
    input_dict = dict.fromkeys(feature_names, 0)

    # Fill numeric inputs
    input_dict["temparature"] = temperature
    input_dict["humidity"] = humidity
    input_dict["moisture"] = moisture
    input_dict["nitrogen"] = N
    input_dict["potassium"] = K
    input_dict["phosphorous"] = P

    # One-hot encoding
    soil_key = f"soil_type_{soil}"
    crop_key = f"crop_type_{crop}"
    if soil_key in input_dict:
        input_dict[soil_key] = 1
    if crop_key in input_dict:
        input_dict[crop_key] = 1

    # Predict
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Fertilizer: **{prediction}**")


