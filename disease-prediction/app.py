import os
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, le, metadata


def pretty(symptom: str) -> str:
    return symptom.replace("_", " ").capitalize()


st.title("🩺 Disease Prediction")

if not os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib")):
    st.error("No trained model found. Run `python src/train.py` first, then reload this app.")
    st.stop()

model, le, metadata = load_artifacts()
all_symptoms = metadata["symptom_columns"]

st.subheader("Select your symptoms")
cols = st.columns(2)
selected = []
for i, symptom in enumerate(all_symptoms):
    with cols[i % 2]:
        if st.checkbox(pretty(symptom), key=symptom):
            selected.append(symptom)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        X = pd.DataFrame([[1 if s in selected else 0 for s in all_symptoms]], columns=all_symptoms)
        probs = model.predict_proba(X)[0]
        ranked = sorted(zip(le.classes_, probs), key=lambda t: t[1], reverse=True)

        st.subheader("Top predictions")
        for disease, prob in ranked[:5]:
            st.write(f"**{disease}** — {prob*100:.1f}%")
            st.progress(min(float(prob), 1.0))