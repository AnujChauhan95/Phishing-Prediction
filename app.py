import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Phishing Website Detection", layout="centered")

st.title("🛡️ Phishing Website Detection")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV with features and target column `Result`", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    if "Result" not in df.columns:
        st.error("The dataset must contain a 'Result' column for labels.")
    else:
        # Basic stats
        st.subheader("📊 Dataset Info")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        # Prepare data
        X = df.drop(columns=["Result"])
        y = df["Result"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("✅ Model Evaluation")
        st.write(f"Accuracy on test data: **{acc:.2f}**")

        # Predict with user input
        st.subheader("🧪 Try Model with Custom Input")
        sample_input = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=0)
            sample_input.append(val)

        if st.button("Predict"):
            pred = model.predict([sample_input])
            label = "Legitimate" if pred[0] == 1 else "Phishing"
            st.success(f"Prediction: {label}")
