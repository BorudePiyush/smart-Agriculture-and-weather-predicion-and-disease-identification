import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="SmartFarm Disease Detection", layout="centered")

# Hide Streamlit main menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load model from SavedModel folder (not .keras file)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "trained_plant_disease_model")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error("❌ Failed to load the model. Make sure the folder exists and is valid.")
        st.exception(e)
        return None

# Predict function
def model_prediction(test_image):
    model = load_model()
    if model is None:
        return None
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar navigation
st.sidebar.title("🌾 SmartFarm")
app_mode = st.sidebar.selectbox("Navigate", ["🏠 Home", "🦠 Disease Recognition"])

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🌿 SmartFarm Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👤 Reviewer")
    st.markdown("**Name**: Piyush Borude")
    st.markdown("**Location**: India")
    st.markdown("**Status**: Active profile")

# Disease Detection Page
elif app_mode == "🦠 Disease Recognition":
    st.header("📸 Upload Plant Image")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔍 Predicting disease...")
            result_index = model_prediction(test_image)

            if result_index is None:
                st.error("Prediction failed due to model loading issue.")
            else:
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.success(f"🌱 Model Prediction: **{class_name[result_index]}**")
