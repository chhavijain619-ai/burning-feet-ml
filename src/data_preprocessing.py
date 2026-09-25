"""
data_preprocessing.py
----------------------
Data cleaning, feature engineering, and preprocessing pipeline construction
for the Burning Feet Syndrome Symptom Assessment project.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET_COL = "assessment"

BINARY_SYMPTOM_COLS = [
    "burning_sensation", "tingling", "numbness", "foot_pain",
    "warmth_sensation", "redness", "swelling", "night_time_symptoms",
]

BINARY_HEALTH_COLS = [
    "diabetes_history", "vitamin_deficiency_history", "thyroid_condition",
    "alcohol_risk_factor", "peripheral_neuropathy_history",
]

NUMERIC_COLS = ["age", "duration_weeks", "severity_level"]
CATEGORICAL_COLS = ["gender", "physical_activity"]


def load_data(path="data/burning_feet_dataset.csv"):
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicate records."""
    df = df.copy()

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    # Numeric missing values -> median imputation (robust to outliers)
    for col in ["duration_weeks", "severity_level"]:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical missing values -> mode imputation
    if df["physical_activity"].isnull().sum() > 0:
        mode_val = df["physical_activity"].mode()[0]
        df["physical_activity"] = df["physical_activity"].fillna(mode_val)

    # Guard against impossible values
    df["age"] = df["age"].clip(lower=1, upper=110)
    df["duration_weeks"] = df["duration_weeks"].clip(lower=0, upper=104)
    df["severity_level"] = df["severity_level"].clip(lower=1, upper=10)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that summarize symptom burden and risk."""
    df = df.copy()

    # Symptom Score: sum of all binary symptom indicators (0-8)
    df["symptom_score"] = df[BINARY_SYMPTOM_COLS].sum(axis=1)

    # Risk Factor Count: sum of relevant health-history binary factors (0-5)
    df["risk_factor_count"] = df[BINARY_HEALTH_COLS].sum(axis=1)

    # Duration Category: group symptom duration into meaningful bins
    df["duration_category"] = pd.cut(
        df["duration_weeks"],
        bins=[-0.1, 2, 8, 26, 1000],
        labels=["Acute (<2wks)", "Short-term (2-8wks)", "Chronic (8-26wks)", "Long-term (26wks+)"],
    ).astype(str)

    return df


ENGINEERED_NUMERIC_COLS = ["symptom_score", "risk_factor_count"]
ENGINEERED_CATEGORICAL_COLS = ["duration_category"]


def get_feature_columns():
    numeric = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
    categorical = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS
    binary = BINARY_SYMPTOM_COLS + BINARY_HEALTH_COLS
    return numeric, categorical, binary


def build_preprocessing_pipeline():
    """
    Builds a ColumnTransformer that:
      - Scales numeric columns with StandardScaler
      - One-hot encodes categorical columns
      - Passes binary (already 0/1) columns through, but imputes just in case
    """
    numeric_cols, categorical_cols, binary_cols = get_feature_columns()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("bin", binary_transformer, binary_cols),
    ])

    return preprocessor


def prepare_full_dataframe(path="data/burning_feet_dataset.csv"):
    """Convenience function: load -> clean -> engineer features."""
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = prepare_full_dataframe()
    print(df.head())
    print("\nShape after cleaning + feature engineering:", df.shape)
    print("\nFeature columns used by the model:")
    numeric, categorical, binary = get_feature_columns()
    print("Numeric:", numeric)
    print("Categorical:", categorical)
    print("Binary:", binary)
