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

st.markdown(
    "<h1 style='text-align: center;'>❤️ Heart Disease Classification Dashboard</h1>",
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------
# Dataset Selection
# ----------------------------------
st.subheader("📂 Dataset Selection")

dataset_option = st.radio(
    "Select test data source:",
    ("Use a sample test dataset bundled with the app", "Upload your own CSV file")
)

# ----------------------------------
# Load Dataset
# ----------------------------------
df = None

if dataset_option == "Use a sample test dataset bundled with the app":
    df = pd.read_csv("data/heart_disease_sample_data.csv")
    st.info("Using sample test dataset bundled with the app.")

 # Download link for test dataset
    with open("data/heart_disease_sample_data.csv", "rb") as file:
        st.download_button(
            label="⬇️ Download sample test dataset (heart_disease_sample_data.csv)",
            data=file,
            file_name="heart_disease_sample_data.csv",
            mime="text/csv"
        )

elif dataset_option == "Upload your own CSV file":
    uploaded_file = st.file_uploader(
        "Upload CSV file (must include 'target' column)",
        type=["csv"]
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

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

st.subheader("🔧 Model Selection")
selected_model = st.selectbox(
    "Choose a Machine Learning Model",
    list(model_map.keys())
)


# ----------------------------------
# Validation
# ----------------------------------
if df is not None:
    if "target" not in df.columns:
        st.error("Dataset must contain a 'target' column.")
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
        # Metrics Display
        # ----------------------------------
        st.subheader("📊 Evaluation Metrics")

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
        # Simple Confusion Matrix
        # ----------------------------------

        # Center the box using columns
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    with st.container(border=True):
        sns.set_theme(style="dark")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(3.2, 2.6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="mako",
            cbar=True,                 # ✅ SHOW COLOR METER
            cbar_kws={"shrink": 0.7},  # ✅ MAKE METER SMALL
            linewidths=0.5,
            linecolor="gray",
            xticklabels=["No Disease (0)", "Disease (1)"],
            yticklabels=["No Disease (0)", "Disease (1)"],
            ax=ax
        )

        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

        st.pyplot(fig, use_container_width=False)
