🌾 Smart Agriculture Support System 🌦️🦠
A smart solution to assist farmers with fertilizer recommendations, weather condition monitoring, and crop disease detection using machine learning and web-based technologies.

📌 Project Overview
This project integrates three main functionalities into one intelligent agricultural support system:

Fertilizer Recommendation: Suggests the most suitable fertilizer based on soil nutrients, crop type, and environmental parameters.

Weather Detection: Retrieves real-time or simulated weather data for analysis.

Disease Identification: Uses machine learning to detect crop diseases from leaf images.

🧠 Features
✅ Fertilizer recommendation based on:

Soil Type

Crop Type

NPK (Nitrogen, Phosphorus, Potassium)

Temperature, Humidity, Moisture

✅ Weather support module (optional real-time integration via API)

✅ Plant disease prediction using image classification (CNN)

✅ Built with Python, Streamlit, Pandas, Scikit-learn, and TensorFlow/Keras

🗂️ Project Structure
bash
Copy
Edit
FERTILIZER-RECOMMENDATION/
│
├── fertilizer_app.py              # Streamlit web app for fertilizer recommendation
├── fertilizer_data.csv            # Dataset used for training
├── fertilizer_model_train.py      # Model training script
├── fertilizer_model.pkl           # Trained model
├── requirements.txt               # Python dependencies
│
DISEASE-DETECTION/
├── disease_predict.py             # Streamlit app for disease prediction
├── model/                         # Trained CNN model
│   └── plant_disease_model.h5
├── sample_leaf_images/            # Sample test images
│
WEATHER-MODULE/
├── weather_module.py              # Weather prediction or API connector
🚀 Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/smart-agriculture-support-system.git
cd smart-agriculture-support-system/FERTILIZER-RECOMMENDATION
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Fertilizer App

bash
Copy
Edit
streamlit run fertilizer_app.py
(Repeat for other apps accordingly.)

🧪 Dataset Sources
Fertilizer Dataset: Custom-built or UCI Machine Learning Repository

Disease Dataset: PlantVillage or Kaggle leaf image datasets

Weather Data: OpenWeatherMap API (optional)

📷 Crop Disease Identification (optional extension)
Upload a leaf image

Predict if the crop is healthy or has a specific disease

Model used: CNN trained on PlantVillage dataset

✅ Requirements
Python 3.8+

streamlit

pandas

scikit-learn

numpy

matplotlib

tensorflow / keras (for disease model)

📌 Future Enhancements
Real-time weather API integration

IoT sensor input support

Farmer dashboard with historical analytics

Multi-language support for farmers

