import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ✅ Streamlit Page Config
st.set_page_config(page_title="SmartFarm", page_icon="🌾", layout="centered")

# ✅ Display Header Image
img_path = os.path.join(os.path.dirname(__file__), "crop.png")
if os.path.exists(img_path):
    try:
        img = Image.open(img_path)
        st.write(f"Image type: {type(img)}")  # Debug: show image type
        st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not load image: {e}")
else:
    st.warning("⚠️ Image 'crop.png' not found.")

# ✅ Load CSV dataset
csv_path = os.path.join(os.path.dirname(__file__), 'Crop_recommendation.csv')
if not os.path.exists(csv_path):
    st.error("❌ 'Crop_recommendation.csv' file not found. Please make sure it's in the same folder.")
    st.stop()

df = pd.read_csv(csv_path)

# ✅ Prepare data
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Load or train model safely
model_path = os.path.join(os.path.dirname(__file__), 'RF.pkl')


try:
    with open(model_path, 'rb') as f:
        RF_Model = pickle.load(f)
except Exception as e:
    st.warning(f"⚠️ Model loading failed: {e}. Retraining the model...")
    RF_Model = RandomForestClassifier(n_estimators=20, random_state=5)
    RF_Model.fit(Xtrain, Ytrain)
    with open(model_path, 'wb') as f:
        pickle.dump(RF_Model, f)

# ✅ Prediction Function
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = RF_Model.predict(data)
    return prediction[0]

# ✅ Show Image of Recommended Crop
def show_crop_image(crop_name):
    image_name = crop_name.lower() + ".jpg"
    image_path = os.path.join("crop_images", image_name)
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            st.image(img, caption=f"Recommended Crop: {crop_name}", use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not load crop image: {e}")
    else:
        st.info(f"ℹ️ No image found for '{crop_name}'.")

# ✅ Main UI
def main():
    st.markdown("<h1 style='text-align: center; color: green;'>🌱 SmartFarm: Smart Crop Recommendations</h1>", unsafe_allow_html=True)

    st.sidebar.header("🌾 Input Soil and Weather Conditions")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=0.0, step=0.5)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=0.0, step=0.5)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=0.0, step=0.5)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.5)

    if st.sidebar.button("🌾 Recommend Crop"):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"✅ Based on your input, the recommended crop is: **{prediction.upper()}**")
        show_crop_image(prediction)

    # Optional: View dataset sample
    with st.expander("📊 View Sample Data"):
        st.dataframe(df.head(10))

# ✅ Run App
if __name__ == "__main__":
    main()



  
