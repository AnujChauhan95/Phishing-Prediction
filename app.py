import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Web Phishing Detector", layout="wide")
st.title("🔐 Web Page Phishing Detection")

# Load trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    xgb = joblib.load(model_path)
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ Model file not found. Please upload `model.pkl`.")

# Input form for user
st.subheader("🧪 Test a Webpage by Entering Features")

feature_names = [
    'url_length', 'n_dots', 'n_hypens', 'n_underline',
    'n_slash', 'n_questionmark', 'n_redirection'
]

user_input = []
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    val = cols[i % 3].number_input(
        label=f"{feature}",
        min_value=0,
        value=1,
        step=1
    )
    user_input.append(val)

# Predict
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = xgb.predict(input_array)

    if prediction[0] == 1:
        st.error("⚠️ Phishing Detected!")
    else:
        st.success("✅ Looks Safe!")

