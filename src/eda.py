"""
eda.py
------
Exploratory Data Analysis for the Burning Feet Syndrome dataset.
Generates summary statistics and saves visualizations to reports/figures/.

Run:
    python src/eda.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import prepare_full_dataframe, BINARY_SYMPTOM_COLS

FIG_DIR = "reports/figures"
sns.set_style("whitegrid")


def dataset_overview(df: pd.DataFrame):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nStatistical summary:\n", df.describe(include="all").T)


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    order = df["assessment"].value_counts().index
    sns.countplot(data=df, x="assessment", order=order, palette="viridis", ax=ax)
    ax.set_title("Class Distribution of Assessment Category")
    ax.set_xlabel("Assessment Category")
    ax.set_ylabel("Count")
    save_fig(fig, "01_class_distribution.png")


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["age"], bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Frequency")
    save_fig(fig, "02_age_distribution.png")


def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="gender", palette="pastel", ax=ax)
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    save_fig(fig, "03_gender_distribution.png")


def plot_symptom_distribution(df, col, title, fname):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=col, palette="Set2", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f"{col} (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    save_fig(fig, fname)


def plot_condition_vs_assessment(df, col, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="assessment", hue=col, palette="coolwarm", ax=ax,
                   order=["Low", "Moderate", "High"])
    ax.set_title(title)
    ax.set_xlabel("Assessment Category")
    ax.set_ylabel("Count")
    ax.legend(title=col)
    save_fig(fig, fname)


def plot_severity_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["severity_level"], bins=10, kde=False, color="orange", ax=ax)
    ax.set_title("Symptom Severity Level Distribution")
    ax.set_xlabel("Severity Level (1-10)")
    ax.set_ylabel("Frequency")
    save_fig(fig, "09_severity_distribution.png")


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    save_fig(fig, "10_correlation_heatmap.png")


def run_eda():
    df = prepare_full_dataframe()
    dataset_overview(df)

    plot_class_distribution(df)                                              # 1
    plot_age_distribution(df)                                                # 2
    plot_gender_distribution(df)                                             # 3
    plot_symptom_distribution(df, "burning_sensation",
                               "Burning Sensation Distribution", "04_burning_sensation.png")  # 4
    plot_symptom_distribution(df, "tingling",
                               "Tingling Distribution", "05_tingling.png")    # 5
    plot_symptom_distribution(df, "numbness",
                               "Numbness Distribution", "06_numbness.png")    # 6
    plot_condition_vs_assessment(df, "diabetes_history",
                                  "Diabetes History vs Assessment", "07_diabetes_vs_assessment.png")  # 7
    plot_condition_vs_assessment(df, "vitamin_deficiency_history",
                                  "Vitamin Deficiency vs Assessment", "08_vitamin_vs_assessment.png")  # 8
    plot_severity_distribution(df)                                           # 9
    plot_correlation_heatmap(df)                                             # 10

    print("\nEDA complete. All figures saved to:", FIG_DIR)
    return df


if __name__ == "__main__":
    run_eda()
