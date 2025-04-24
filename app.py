import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load model
model = joblib.load("xgb_model.pkl")

st.title("Phishing Website Detector")

# Input fields
n_at = st.number_input("Number of @ symbols", value=0.0)
n_tilde = st.number_input("Number of ~ symbols", value=0.0)
n_redirection = st.number_input("Number of redirections", value=0.0)

# Add more features if needed...

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "n_at": n_at,
        "n_tilde": n_tilde,
        "n_redirection": n_redirection
        # Add more keys if additional features are used
    }])
    prediction = model.predict(input_df)
    result = "Phishing Website" if prediction[0] == 1 else "Legitimate Website"
    st.success(f"Prediction: {result}")
