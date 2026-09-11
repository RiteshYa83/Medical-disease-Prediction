# Disease Prediction from Symptoms — ML Project

A complete, end-to-end machine learning project that predicts the most likely
disease given a set of reported symptoms. Built as a learning/demo project —
**not a real diagnostic tool.**

## How it works

1. **Data** (`data/generate_data.py`) — generates a synthetic dataset of
   patients, each labeled with one of 14 common conditions (flu, diabetes,
   migraine, asthma, malaria, etc.), with symptoms sampled according to
   realistic disease-symptom probabilities plus background noise. Swap this
   out for a real dataset later (see "Using real data" below).
2. **Training** (`src/train.py`) — trains and compares 5 classifiers
   (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN) using
   train/test split + 5-fold cross-validation, then saves the best model.
3. **Prediction** — get predictions either via:
   - CLI: `src/predict.py`
   - Web UI: `app.py` (Streamlit)

## Project structure

```
disease-prediction/
├── data/
│   ├── generate_data.py      # creates the synthetic dataset
│   └── disease_data.csv      # generated dataset (symptoms + disease label)
├── src/
│   ├── train.py               # trains models, saves the best one
│   └── predict.py             # CLI prediction tool
├── models/                    # created after training
│   ├── best_model.joblib
│   ├── label_encoder.joblib
│   └── metadata.json          # symptom list, class list, performance metrics
├── app.py                     # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset
```bash
python data/generate_data.py
```

### 2. Train models
```bash
python src/train.py
```
This prints a comparison table across all 5 models and saves the best one
(picked by cross-validated accuracy) to `models/`.

### 3. Predict from the command line
```bash
python src/predict.py --list-symptoms
python src/predict.py --symptoms fever chills muscle_pain sweating headache
```

### 4. Or use the web app
```bash
streamlit run app.py
```
Check symptom boxes and click **Predict** to see ranked disease probabilities.

## Current model performance

With the synthetic dataset, Logistic Regression is typically the best
performer at ~87-88% cross-validated accuracy across 14 disease classes.
Diseases with very distinctive symptom signatures (Diabetes, Hepatitis,
Tuberculosis) predict with >90% precision/recall; diseases that share
overlapping symptoms with others (Flu vs. Typhoid vs. Malaria, all
fever-driven) are harder to separate — which mirrors real-world diagnostic
difficulty and is a good discussion point if this is for a class project.

## Using real data

To move beyond the synthetic demo, replace `data/disease_data.csv` with a
real dataset in the same shape: one column per binary symptom (0/1), plus a
`disease` column with the diagnosis label. A well-known public option is the
"Disease Prediction Using Machine Learning" symptom-checker dataset on
Kaggle. `src/train.py` and `src/predict.py` require no changes — they infer
the symptom list and class list directly from the CSV columns.

## Extending this project

- **Add explainability**: use SHAP or permutation importance to show which
  symptoms drove a given prediction.
- **Add severity/demographics**: extend features beyond binary symptoms
  (age, duration of symptoms, severity scale) for a richer model.
- **Hyperparameter tuning**: wrap `GridSearchCV`/`RandomizedSearchCV` around
  the candidate models in `train.py`.
- **Multi-label setting**: real patients can have co-occurring conditions —
  consider `MultiOutputClassifier` if that's relevant to your use case.
- **Deploy**: wrap `predict.py`'s logic in a Flask/FastAPI endpoint for a
  production-style API, or deploy the Streamlit app to Streamlit Community
  Cloud.

## ⚠️ Important disclaimer

This project is for **educational purposes only**. The dataset is
synthetic, the model is not clinically validated, and predictions must
never be used for actual medical diagnosis or treatment decisions. Anyone
with health concerns should consult a qualified healthcare professional.
