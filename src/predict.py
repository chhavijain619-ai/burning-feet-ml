"""
predict.py
----------
Shared inference module used by both the Streamlit application and the test
suite. Loads the saved pipeline once, builds the engineered features for a
single raw user input, and returns a preliminary ML-based assessment.

IMPORTANT: This produces an academic, ML-based PRELIMINARY ASSESSMENT only.
It is NOT a medical diagnosis.
"""

import os
import json
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "burning_feet_model.pkl")
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model_metadata.json")

_pipeline = None
_metadata = None


def load_artifacts():
    """Load (and cache) the trained pipeline and metadata."""
    global _pipeline, _metadata
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python src/train_models.py` first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _pipeline, _metadata


def build_input_dataframe(raw_input: dict) -> pd.DataFrame:
    """
    Takes a dict of raw user-entered values (matching the ORIGINAL dataset
    columns, before feature engineering) and returns a single-row DataFrame
    with the engineered features the pipeline expects.

    Expected keys in raw_input:
        age, gender, burning_sensation, tingling, numbness, foot_pain,
        warmth_sensation, redness, swelling, night_time_symptoms,
        duration_weeks, diabetes_history, vitamin_deficiency_history,
        thyroid_condition, alcohol_risk_factor, peripheral_neuropathy_history,
        physical_activity, severity_level
    """
    df = pd.DataFrame([raw_input])

    binary_symptom_cols = [
        "burning_sensation", "tingling", "numbness", "foot_pain",
        "warmth_sensation", "redness", "swelling", "night_time_symptoms",
    ]
    binary_health_cols = [
        "diabetes_history", "vitamin_deficiency_history", "thyroid_condition",
        "alcohol_risk_factor", "peripheral_neuropathy_history",
    ]

    df["symptom_score"] = df[binary_symptom_cols].sum(axis=1)
    df["risk_factor_count"] = df[binary_health_cols].sum(axis=1)
    df["duration_category"] = pd.cut(
        df["duration_weeks"],
        bins=[-0.1, 2, 8, 26, 1000],
        labels=["Acute (<2wks)", "Short-term (2-8wks)", "Chronic (8-26wks)", "Long-term (26wks+)"],
    ).astype(str)

    return df


def predict_assessment(raw_input: dict):
    """
    Runs the full pipeline on a single raw input and returns:
        {
            "prediction": "Low" | "Moderate" | "High",
            "probabilities": {class_name: probability, ...} or None,
            "model_name": str,
        }
    """
    pipeline, metadata = load_artifacts()
    X = build_input_dataframe(raw_input)

    prediction = pipeline.predict(X)[0]

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)[0]
            classes = pipeline.classes_
            probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
        except Exception:
            probabilities = None

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "model_name": metadata.get("model_name", "Unknown"),
    }


if __name__ == "__main__":
    sample_input = {
        "age": 55,
        "gender": "Male",
        "burning_sensation": 1,
        "tingling": 1,
        "numbness": 1,
        "foot_pain": 1,
        "warmth_sensation": 1,
        "redness": 0,
        "swelling": 0,
        "night_time_symptoms": 1,
        "duration_weeks": 10,
        "diabetes_history": 1,
        "vitamin_deficiency_history": 0,
        "thyroid_condition": 0,
        "alcohol_risk_factor": 0,
        "peripheral_neuropathy_history": 1,
        "physical_activity": "Low",
        "severity_level": 7,
    }
    result = predict_assessment(sample_input)
    print("Sample prediction result:", result)
