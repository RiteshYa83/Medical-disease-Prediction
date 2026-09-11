"""
predict.py
----------
Command-line tool to get a disease prediction from a list of symptoms.

Usage examples:
    python src/predict.py --list-symptoms
    python src/predict.py --symptoms fever chills headache muscle_pain
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    for p in (model_path, le_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run `python src/train.py` first to train and save a model."
            )

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, le, metadata


def symptoms_to_vector(selected_symptoms, all_symptoms):
    unknown = [s for s in selected_symptoms if s not in all_symptoms]
    if unknown:
        raise ValueError(
            f"Unknown symptom(s): {unknown}. Use --list-symptoms to see valid options."
        )
    vector = [1 if s in selected_symptoms else 0 for s in all_symptoms]
    return pd.DataFrame([vector], columns=all_symptoms)


def predict(selected_symptoms, top_k=3):
    model, le, metadata = load_artifacts()
    all_symptoms = metadata["symptom_columns"]

    X = symptoms_to_vector(selected_symptoms, all_symptoms)
    probs = model.predict_proba(X)[0]

    ranked = sorted(zip(le.classes_, probs), key=lambda t: t[1], reverse=True)
    return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Predict likely disease from symptoms.")
    parser.add_argument("--symptoms", nargs="+", help="List of symptom names (snake_case).")
    parser.add_argument("--list-symptoms", action="store_true", help="Print all valid symptom names.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show.")
    args = parser.parse_args()

    if args.list_symptoms:
        _, _, metadata = load_artifacts()
        print("Valid symptoms:")
        for s in metadata["symptom_columns"]:
            print(f"  - {s}")
        return

    if not args.symptoms:
        parser.error("Provide --symptoms (or use --list-symptoms to see valid options).")

    results = predict(args.symptoms, top_k=args.top_k)

    print(f"\nInput symptoms: {', '.join(args.symptoms)}")
    print("\nTop predictions:")
    for disease, prob in results:
        print(f"  {disease:20s} {prob*100:5.1f}%")

    print(
        "\nNote: this is an educational demo model trained on synthetic data. "
        "It is NOT a medical device and should never be used for real diagnosis. "
        "Always consult a qualified healthcare professional."
    )


if __name__ == "__main__":
    main()
