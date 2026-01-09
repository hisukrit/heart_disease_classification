import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

# ----------------------------------
# Header
# ----------------------------------
st.markdown(
    "<h1 style='text-align: center;'>❤️ Heart Disease Classification Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Upload test data, select a model, and evaluate performance</p>",
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------
# Model Selection
# ----------------------------------
model_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

st.subheader("🔧 Model Configuration")
selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_map.keys())
)

# ----------------------------------
# Upload CSV
# ----------------------------------
st.subheader("📂 Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV file (must include 'target' column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X = df.drop("target", axis=1)
        y = df["target"]

        model = joblib.load(model_map[selected_model])

        # Predictions
        y_pred = model.predict(X)

        # AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = np.nan

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.divider()

        # ----------------------------------
        # Metric Cards
        # ----------------------------------
        st.subheader("📊 Model Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("AUC Score", f"{auc:.3f}")
        col3.metric("Precision", f"{precision:.3f}")
        col4.metric("Recall", f"{recall:.3f}")
        col5.metric("F1 Score", f"{f1:.3f}")
        col6.metric("MCC", f"{mcc:.3f}")

        st.divider()

     # ----------------------------------
    # Confusion Matrix
    # ----------------------------------
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Disease (0)", "Actual: Disease (1)"],
        columns=["Predicted: No Disease (0)", "Predicted: Disease (1)"]
    )

    st.table(cm_df)
