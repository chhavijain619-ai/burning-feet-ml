"""
test_prediction.py
-------------------
Test suite for the Burning Feet Syndrome prediction pipeline.

Run:
    pytest tests/test_prediction.py -v
"""

import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_assessment, load_artifacts, build_input_dataframe

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "burning_feet_model.pkl")


def base_valid_input(**overrides):
    payload = {
        "age": 45,
        "gender": "Male",
        "burning_sensation": 1,
        "tingling": 0,
        "numbness": 0,
        "foot_pain": 1,
        "warmth_sensation": 0,
        "redness": 0,
        "swelling": 0,
        "night_time_symptoms": 0,
        "duration_weeks": 5,
        "diabetes_history": 0,
        "vitamin_deficiency_history": 0,
        "thyroid_condition": 0,
        "alcohol_risk_factor": 0,
        "peripheral_neuropathy_history": 0,
        "physical_activity": "Moderate",
        "severity_level": 4,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Trained model file is missing. Run src/train_models.py."


def test_model_loading():
    pipeline, metadata = load_artifacts()
    assert pipeline is not None
    assert "model_name" in metadata
    assert "target_classes" in metadata


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------

def test_valid_input_returns_valid_class():
    result = predict_assessment(base_valid_input())
    assert result["prediction"] in ["Low", "Moderate", "High"]


def test_valid_input_returns_probabilities_summing_to_one():
    result = predict_assessment(base_valid_input())
    if result["probabilities"] is not None:
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Edge cases: minimum / maximum age
# ---------------------------------------------------------------------------

def test_minimum_age():
    result = predict_assessment(base_valid_input(age=1))
    assert result["prediction"] in ["Low", "Moderate", "High"]


def test_maximum_age():
    result = predict_assessment(base_valid_input(age=110))
    assert result["prediction"] in ["Low", "Moderate", "High"]


# ---------------------------------------------------------------------------
# High-risk vs low-risk symptom combinations
# ---------------------------------------------------------------------------

def test_high_risk_combination_runs():
    high_risk_input = base_valid_input(
        burning_sensation=1, tingling=1, numbness=1, foot_pain=1,
        warmth_sensation=1, night_time_symptoms=1, duration_weeks=30,
        diabetes_history=1, peripheral_neuropathy_history=1, severity_level=9,
        physical_activity="Low",
    )
    result = predict_assessment(high_risk_input)
    assert result["prediction"] in ["Low", "Moderate", "High"]


def test_low_risk_combination_runs():
    low_risk_input = base_valid_input(
        burning_sensation=0, tingling=0, numbness=0, foot_pain=0,
        warmth_sensation=0, night_time_symptoms=0, duration_weeks=0.5,
        diabetes_history=0, peripheral_neuropathy_history=0, severity_level=1,
        physical_activity="High",
    )
    result = predict_assessment(low_risk_input)
    assert result["prediction"] in ["Low", "Moderate", "High"]


# ---------------------------------------------------------------------------
# Missing / invalid input handling
# ---------------------------------------------------------------------------

def test_missing_optional_field_raises_or_handled():
    incomplete_input = base_valid_input()
    del incomplete_input["swelling"]
    with pytest.raises(KeyError):
        build_input_dataframe(incomplete_input)


def test_invalid_gender_value_does_not_crash_pipeline():
    # Unknown categories are handled by OneHotEncoder(handle_unknown="ignore")
    weird_input = base_valid_input(gender="Unknown")
    result = predict_assessment(weird_input)
    assert result["prediction"] in ["Low", "Moderate", "High"]


def test_invalid_physical_activity_value_does_not_crash_pipeline():
    weird_input = base_valid_input(physical_activity="Extreme")
    result = predict_assessment(weird_input)
    assert result["prediction"] in ["Low", "Moderate", "High"]


# ---------------------------------------------------------------------------
# build_input_dataframe feature engineering correctness
# ---------------------------------------------------------------------------

def test_symptom_score_calculation():
    payload = base_valid_input(
        burning_sensation=1, tingling=1, numbness=1, foot_pain=0,
        warmth_sensation=0, redness=0, swelling=0, night_time_symptoms=0,
    )
    df = build_input_dataframe(payload)
    assert df.loc[0, "symptom_score"] == 3


def test_risk_factor_count_calculation():
    payload = base_valid_input(
        diabetes_history=1, vitamin_deficiency_history=1, thyroid_condition=0,
        alcohol_risk_factor=0, peripheral_neuropathy_history=1,
    )
    df = build_input_dataframe(payload)
    assert df.loc[0, "risk_factor_count"] == 3


def test_duration_category_binning():
    payload = base_valid_input(duration_weeks=1)
    df = build_input_dataframe(payload)
    assert df.loc[0, "duration_category"] == "Acute (<2wks)"
