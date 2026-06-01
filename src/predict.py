"""Prediction helpers for loading model artifacts and scoring one student row."""

import pickle

import pandas as pd

from .utils import model_path


def load_models():
    """Load regression model, classification model, and metadata."""
    with model_path("student_reg_model.pkl").open("rb") as f:
        reg_model = pickle.load(f)
    with model_path("student_clf_model.pkl").open("rb") as f:
        clf_model = pickle.load(f)

    metadata_path = model_path("student_model_metadata.pkl")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("rb") as f:
            metadata = pickle.load(f)

    return reg_model, clf_model, metadata


def predict_student(input_data: dict) -> dict:
    """Predict final score, pass/fail class, and pass probability for one row."""
    reg_model, clf_model, metadata = load_models()
    input_df = pd.DataFrame([input_data])

    feature_columns = metadata.get("feature_columns")
    if feature_columns:
        input_df = input_df.reindex(columns=feature_columns)

    score = float(reg_model.predict(input_df)[0])
    score = max(0.0, min(20.0, score))
    passing = int(clf_model.predict(input_df)[0])

    probability = None
    if hasattr(clf_model, "predict_proba"):
        probability = float(clf_model.predict_proba(input_df)[0][1])

    return {
        "predicted_score": score,
        "passing": passing,
        "pass_probability": probability,
    }
