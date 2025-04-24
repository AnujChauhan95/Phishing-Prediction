import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
import os

st.set_page_config(page_title="Web Phishing Detector", layout="wide")
st.title("🔐 Web Page Phishing Detection")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Data Preview")
    st.write(df.head())

    st.subheader("🔍 Data Info")
    st.write(df.info())

    # Fill missing values
    cat_col = ['n_at','n_tilde','n_redirection']
    for i in cat_col:
        df[i] = df[i].fillna(df[i].median())

    # Features and label
    X = df.loc[:, ['url_length', 'n_dots', 'n_hypens', 'n_underline', 'n_slash',
                   'n_questionmark', 'n_redirection']]
    Y = df['phishing']

    # Train/test split
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=67, stratify=Y)

    # Train or load model
    model_path = "model.pkl"
    if os.path.exists(model_path):
        xgb = joblib.load(model_path)
        st.success("✅ Loaded existing model.")
    else:
        st.info("⏳ Training XGBoost model...")
        params = {
            "learning_rate": [0.05, 0.10, 0.15, 0.20],
            "max_depth": [4, 6, 8, 10],
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.1, 0.3],
            "colsample_bytree": [0.3, 0.5, 0.7]
        }
        grcv = RandomizedSearchCV(XGBClassifier(random_state=32), params, n_jobs=-1, cv=3)
        grcv.fit(X, Y)
        xgb = grcv.best_estimator_
        joblib.dump(xgb, model_path)
        st.success("✅ Model trained and saved.")

    # Evaluate
    xgb_pred = xgb.predict(xtest)
    acc = accuracy_score(ytest, xgb_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(ytest, xgb_pred))

    # Prediction form
    st.subheader("🧪 Test Your Own Data")

    input_dict = {}
    for col in ['url_length', 'n_dots', 'n_hypens', 'n_underline', 'n_slash', 'n_questionmark', 'n_redirection']:
        input_dict[col] = st.number_input(f"{col}", min_value=0.0, value=1.0)

    if st.button("Predict"):
        test_input = np.array(list(input_dict.values())).reshape(1, -1)
        result = xgb.predict(test_input)
        if result[0] == 1:
            st.error("⚠️ This webpage is likely a phishing site!")
        else:
            st.success("✅ This webpage appears to be safe.")

