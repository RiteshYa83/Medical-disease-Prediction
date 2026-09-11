"""
train.py
--------
Loads the symptom-disease dataset, trains several classifiers,
compares them, and saves the best-performing model + metadata
to the models/ directory for later use by predict.py or app.py.

Usage:
    python src/train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "disease_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run data/generate_data.py first "
            "(or replace it with your own real dataset in the same format: "
            "binary symptom columns + a 'disease' target column)."
        )
    df = pd.read_csv(DATA_PATH)
    symptom_cols = [c for c in df.columns if c != "disease"]
    X = df[symptom_cols]
    y = df["disease"]
    return X, y, symptom_cols


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def main():
    print("Loading data...")
    X, y, symptom_cols = load_data()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    models = get_candidate_models()
    results = {}
    trained_models = {}

    print(f"\nTraining {len(models)} candidate models on {len(X_train)} samples...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv_scores = cross_val_score(model, X, y_enc, cv=5)

        results[name] = {
            "test_accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
        }
        trained_models[name] = model

        print(f"{name:20s} | test_acc={acc:.4f} | macro_f1={f1:.4f} "
              f"| cv_acc={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # pick best by cross-validated accuracy (more robust than a single test split)
    best_name = max(results, key=lambda n: results[n]["cv_mean_accuracy"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    # detailed report for the best model
    best_preds = best_model.predict(X_test)
    report = classification_report(
        y_test, best_preds, target_names=le.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, best_preds).tolist()

    # Save artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    metadata = {
        "best_model": best_name,
        "symptom_columns": symptom_cols,
        "disease_classes": le.classes_.tolist(),
        "model_comparison": results,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model, label encoder, and metadata to {MODELS_DIR}/")
    print("\nPer-class performance (best model):")
    for cls in le.classes_:
        stats = report[cls]
        print(f"  {cls:20s} precision={stats['precision']:.2f} "
              f"recall={stats['recall']:.2f} f1={stats['f1-score']:.2f}")


if __name__ == "__main__":
    main()
