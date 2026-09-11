"""
generate_data.py
-----------------
Generates a synthetic-but-realistic symptom -> disease dataset.

Each disease has a characteristic set of symptoms that occur with high
probability, plus some background noise symptoms that occur occasionally
(to mimic real-world overlap between conditions). This produces a dataset
that is learnable but not trivial, similar in spirit to public
"symptom checker" datasets.

Run directly to (re)create data/disease_data.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES_PER_DISEASE = 300

# Master symptom list
SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "sore_throat", "runny_nose",
    "shortness_of_breath", "chest_pain", "nausea", "vomiting", "diarrhea",
    "abdominal_pain", "joint_pain", "muscle_pain", "rash", "chills",
    "sweating", "weight_loss", "increased_thirst", "frequent_urination",
    "blurred_vision", "dizziness", "high_blood_pressure_history",
    "loss_of_appetite", "yellow_skin", "dark_urine", "sensitivity_to_light",
    "night_sweats", "wheezing", "swollen_lymph_nodes",
]

# Disease -> {symptom: probability of occurring if patient has this disease}
# Includes "signature" symptoms (high prob) and a few overlapping ones (mid prob)
DISEASE_PROFILES = {
    "Common Cold": {
        "runny_nose": 0.9, "sore_throat": 0.75, "cough": 0.7, "sneezing_like_fatigue": 0.0,
        "headache": 0.4, "fever": 0.3, "fatigue": 0.4, "chills": 0.2,
    },
    "Influenza (Flu)": {
        "fever": 0.9, "chills": 0.8, "muscle_pain": 0.85, "fatigue": 0.85,
        "headache": 0.7, "cough": 0.6, "sore_throat": 0.5, "sweating": 0.4,
    },
    "Migraine": {
        "headache": 0.95, "sensitivity_to_light": 0.8, "nausea": 0.6,
        "vomiting": 0.3, "dizziness": 0.4, "fatigue": 0.3,
    },
    "Diabetes": {
        "increased_thirst": 0.9, "frequent_urination": 0.9, "fatigue": 0.6,
        "blurred_vision": 0.55, "weight_loss": 0.4, "loss_of_appetite": 0.2,
    },
    "Hypertension": {
        "headache": 0.5, "dizziness": 0.5, "high_blood_pressure_history": 0.95,
        "chest_pain": 0.25, "shortness_of_breath": 0.2, "blurred_vision": 0.2,
    },
    "Asthma": {
        "shortness_of_breath": 0.9, "wheezing": 0.85, "chest_pain": 0.4,
        "cough": 0.6, "fatigue": 0.3,
    },
    "Pneumonia": {
        "fever": 0.85, "cough": 0.85, "shortness_of_breath": 0.75,
        "chest_pain": 0.6, "fatigue": 0.6, "chills": 0.5, "sweating": 0.3,
    },
    "Heart Disease": {
        "chest_pain": 0.85, "shortness_of_breath": 0.65, "fatigue": 0.55,
        "dizziness": 0.4, "sweating": 0.35, "high_blood_pressure_history": 0.4,
    },
    "Typhoid": {
        "fever": 0.9, "abdominal_pain": 0.6, "loss_of_appetite": 0.6,
        "headache": 0.5, "fatigue": 0.55, "diarrhea": 0.35, "rash": 0.15,
    },
    "Malaria": {
        "fever": 0.95, "chills": 0.85, "sweating": 0.75, "headache": 0.6,
        "muscle_pain": 0.5, "nausea": 0.4, "fatigue": 0.5,
    },
    "Gastroenteritis": {
        "diarrhea": 0.9, "vomiting": 0.7, "abdominal_pain": 0.75,
        "nausea": 0.7, "fever": 0.3, "fatigue": 0.35,
    },
    "Hepatitis": {
        "yellow_skin": 0.8, "dark_urine": 0.7, "fatigue": 0.65,
        "loss_of_appetite": 0.6, "abdominal_pain": 0.5, "nausea": 0.45,
    },
    "Allergy": {
        "rash": 0.7, "runny_nose": 0.5, "sensitivity_to_light": 0.15,
        "swollen_lymph_nodes": 0.1, "cough": 0.2, "sore_throat": 0.15,
    },
    "Tuberculosis": {
        "cough": 0.85, "night_sweats": 0.75, "weight_loss": 0.7,
        "fever": 0.6, "fatigue": 0.6, "chest_pain": 0.35, "loss_of_appetite": 0.4,
    },
}

BACKGROUND_NOISE_PROB = 0.04  # chance any unrelated symptom appears anyway


def generate_dataset(seed=RANDOM_SEED, n_per_disease=N_SAMPLES_PER_DISEASE):
    rng = np.random.default_rng(seed)
    rows = []
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(n_per_disease):
            row = {s: 0 for s in SYMPTOMS}
            for symptom, prob in profile.items():
                if symptom in row and rng.random() < prob:
                    row[symptom] = 1
            # background noise: small chance of an unrelated symptom
            for symptom in SYMPTOMS:
                if symptom not in profile and rng.random() < BACKGROUND_NOISE_PROB:
                    row[symptom] = 1
            row["disease"] = disease
            rows.append(row)

    df = pd.DataFrame(rows)
    # ensure column order: symptoms then target
    df = df[SYMPTOMS + ["disease"]]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "disease_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {df['disease'].nunique()} diseases")
    print(f"Saved to {out_path}")
    print(df["disease"].value_counts())
