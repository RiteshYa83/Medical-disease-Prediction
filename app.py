import streamlit as st
import pickle
import numpy as np

# Load model
with open("disease_model.pkl", "rb") as file:
    model, encoder, symptoms = pickle.load(file)

st.title("🩺 Medical Disease Prediction System")

st.write("Select the symptoms you are experiencing.")

# Create symptom inputs
input_data = []

for symptom in symptoms:
    value = st.checkbox(symptom)
    input_data.append(1 if value else 0)

if st.button("Predict Disease"):

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)

    disease = encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Disease: {disease}")

    st.warning(
        "This prediction is for educational purposes only. "
        "It is not a medical diagnosis."
    )