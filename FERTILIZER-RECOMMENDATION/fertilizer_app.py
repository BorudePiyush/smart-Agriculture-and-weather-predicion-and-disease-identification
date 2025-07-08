import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# App config
st.set_page_config(page_title="🌿 Fertilizer Recommender", layout="centered")
st.title("🌿 Fertilizer Recommendation System")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("fertilizer_data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Train model
@st.cache_resource
def train_model(df):
    df_encoded = pd.get_dummies(df, columns=["soil_type", "crop_type"])
    X = df_encoded.drop("fertilizer_name", axis=1)
    y = df_encoded["fertilizer_name"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

# User Inputs
st.header("🧪 Enter Soil and Crop Details")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=40)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=41)
temperature = st.slider("🌡️ Temperature (°C)", min_value=10, max_value=50, value=25)
humidity = st.slider("💧 Humidity (%)", min_value=0, max_value=100, value=70)
moisture = st.slider("🌾 Soil Moisture (%)", min_value=0, max_value=100, value=30)

soil_options = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
crop_options = [
    "Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
    "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"
]

soil = st.selectbox("🧱 Soil Type", soil_options)
crop = st.selectbox("🌿 Crop Type", crop_options)

# Load model
df = load_data()
model, feature_names = train_model(df)

# Predict on click
if st.button("🔍 Recommend Fertilizer"):
    input_dict = dict.fromkeys(feature_names, 0)

    # Numeric inputs
    input_dict["temparature"] = temperature
    input_dict["humidity"] = humidity
    input_dict["moisture"] = moisture
    input_dict["nitrogen"] = N
    input_dict["potassium"] = K
    input_dict["phosphorous"] = P

    # One-hot encoding for soil and crop
    soil_col = f"soil_type_{soil.lower()}"
    crop_col = f"crop_type_{crop.lower().replace(' ', '_')}"
    if soil_col in input_dict:
        input_dict[soil_col] = 1
    if crop_col in input_dict:
        input_dict[crop_col] = 1

    # Prediction
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Recommended Fertilizer: **{prediction}**")
