import streamlit as st
import numpy as np
import joblib
import os

# -------------------
# Credentials
# -------------------
USERNAME = "admin"
PASSWORD = "password1234"

# -------------------
# Session state check
# -------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# -------------------
# Login Function
# -------------------
def login(username_input, password_input):
    if username_input == USERNAME and password_input == PASSWORD:
        st.session_state.authenticated = True
        st.success("✅ Login successful!")
    else:
        st.error("❌ Invalid username or password")

# -------------------
# Login Page
# -------------------
def login_page():
    st.title("🔐 Login")
    st.write("Please log in to access the phishing detector.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            login(username, password)

# -------------------
# Prediction Page
# -------------------
def prediction_page():
    st.title("🛡️ Web Page Phishing Detector")

    # Load model
    model_path = "model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Model loaded.")
    else:
        st.error("❌ `model.pkl` not found. Please upload the trained model.")
        st.stop()

    st.subheader("🔍 Enter Features to Predict")

    features = [
        'url_length', 'n_dots', 'n_hypens', 'n_underline',
        'n_slash', 'n_questionmark', 'n_redirection'
    ]

    input_vals = []
    cols = st.columns(3)

    for i, feature in enumerate(features):
        val = cols[i % 3].number_input(f"{feature}", min_value=0, value=1)
        input_vals.append(val)

    if st.button("Predict"):
        input_array = np.array(input_vals).reshape(1, -1)
        prediction = model.predict(input_array)

        if prediction[0] == 1:
            st.error("⚠️ This is likely a phishing website!")
        else:
            st.success("✅ This website appears safe.")

# -------------------
# Main App Logic
# -------------------
if not st.session_state.authenticated:
    login_page()
else:
    prediction_page()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by Anuj Chauhan</p>", unsafe_allow_html=True)
