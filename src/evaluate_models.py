"""
evaluate_models.py
-------------------
Loads the saved final pipeline and re-evaluates it on a fresh train/test split
of the dataset (useful for quick sanity-checks after training, or for grading /
viva demonstration without re-running the full training + tuning process).

Run:
    python src/evaluate_models.py
"""

import os
import sys
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import prepare_full_dataframe, get_feature_columns, TARGET_COL

MODEL_PATH = "models/burning_feet_model.pkl"
METADATA_PATH = "models/model_metadata.json"
RANDOM_STATE = 42


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python src/train_models.py` first."
        )

    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    print(f"Loaded model: {metadata['model_name']}")
    print(f"Disclaimer: {metadata['disclaimer']}\n")

    df = prepare_full_dataframe()
    numeric, categorical, binary = get_feature_columns()
    feature_cols = numeric + categorical + binary
    X = df[feature_cols]
    y = df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    y_pred = pipeline.predict(X_test)

    print("Evaluation on held-out test split:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1-score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted; order=Low, Moderate, High):")
    print(confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
