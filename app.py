import json
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Explorer", layout="wide")

# safer root path for streamlit cloud
ROOT = Path(os.getcwd())

RESULTS_PATH = ROOT / "reports" / "models_all" / "results_all.json"
MODELS_DIR = ROOT / "reports" / "models_all" / "models"
ROC_DIR = ROOT / "reports" / "models_all"


def interpret_mcc(mcc_value):
    if mcc_value >= 0.8:
        return "Excellent agreement"
    elif mcc_value >= 0.6:
        return "Strong agreement"
    elif mcc_value >= 0.4:
        return "Moderate agreement"
    elif mcc_value >= 0.2:
        return "Fair agreement"
    elif mcc_value >= 0:
        return "Slight agreement"
    else:
        return "Poor agreement"


@st.cache_data
def load_results():
    try:
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Error reading results file: {e}")
        return {}


@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


def plot_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def main():
    st.title("Bank Marketing Model Explorer")

    results = load_results()

    if not results:
        st.warning("⚠ No model results found yet.")
        st.info("Upload results_all.json inside reports/models_all/")
        st.stop()

    model_names = list(results.keys())
    selected = st.sidebar.selectbox("Select model", model_names)

    metrics = results[selected]
    st.header(selected)

    st.subheader("Metrics")
    st.json(metrics)

    if "mcc" in metrics:
        st.success(f"MCC: {metrics['mcc']} - {interpret_mcc(metrics['mcc'])}")

    if "confusion_matrix" in metrics:
        st.subheader("Confusion Matrix")
        fig = plot_confusion(metrics["confusion_matrix"])
        st.pyplot(fig)

    roc_path = ROC_DIR / f"roc_{selected}.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curve")

    model_file = MODELS_DIR / f"{selected}.joblib"
    if model_file.exists():
        uploaded = st.file_uploader("Upload CSV for prediction")

        if uploaded:
            df = pd.read_csv(uploaded)
            model = load_model(model_file)

            if model:
                preds = model.predict(df)
                st.write(preds[:20])


if __name__ == "__main__":
    main()
