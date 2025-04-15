import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Calories Burned Predictor", page_icon="🔥")

# --- Background Color and Styling (Neon Theme Optional) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #39ff14;
    }
    h1, h2, h3, .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: #39ff14 !important;
        text-shadow: 0 0 5px #39ff14;
    }
    .stButton > button {
        background-color: #000;
        color: #39ff14;
        border: 1px solid #39ff14;
        box-shadow: 0 0 5px #39ff14;
    }
    .stButton > button:hover {
        background-color: #39ff14;
        color: #000;
        box-shadow: 0 0 20px #39ff14;
    }
    .stImage img {
        border-radius: 15px;
        box-shadow: 0 0 20px #39ff14;
    }
    </style>
""", unsafe_allow_html=True)

# --- Top Image ---
st.image("calories_brun_image.webp", use_container_width=True)

# --- Title and Description ---
st.title("🔥 Calories Burned Prediction App")
st.markdown("Enter exercise and physical stats to estimate calories burned.")

# --- Load Model ---
try:
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.stop()

# --- Input Form ---
gender = st.selectbox("Gender", options=["male", "female"])
age = st.number_input("Age (years)", min_value=1, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
duration = st.number_input("Duration of Exercise (minutes)", min_value=0.0, value=30.0)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=250, value=120)
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)

# --- DataFrame Creation ---
input_df = pd.DataFrame([{
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Duration': duration,
    'Heart_Rate': heart_rate,
    'Body_Temp': body_temp
}])

# --- Prediction ---
if st.button("Predict Calories Burned"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🔥 Estimated Calories Burned: **{prediction[0]:.2f} kcal**")
        st.info("✅ Prediction successful!")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --- Footer ---
st.markdown("""
    <hr style="border: none; border-top: 1px solid #39ff14;" />
    <div style='text-align: center; font-weight: bold; color: #39ff14; text-shadow: 0 0 10px #39ff14; font-size: 18px;'>
        Developed by Yash Sharma
    </div>
""", unsafe_allow_html=True)
