"""
train_models.py
----------------
Trains multiple classification models on the Burning Feet Syndrome dataset,
evaluates them, performs hyperparameter tuning on the best candidate,
and saves the final pipeline (preprocessing + model) to models/burning_feet_model.pkl

Run:
    python src/train_models.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import (
    prepare_full_dataframe, build_preprocessing_pipeline, get_feature_columns, TARGET_COL,
)

RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def get_X_y(df):
    numeric, categorical, binary = get_feature_columns()
    feature_cols = numeric + categorical + binary
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_and_compare():
    print("Loading and preparing data...")
    df = prepare_full_dataframe()
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessor = build_preprocessing_pipeline()
    models = get_candidate_models()

    results = []
    fitted_pipelines = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        # Stratified cross-validation on training data
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")

        # Fit on full training set, evaluate on held-out test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics["cv_mean_accuracy"] = cv_scores.mean()
        metrics["cv_std_accuracy"] = cv_scores.std()
        metrics["model"] = name

        results.append(metrics)
        fitted_pipelines[name] = pipe

        print(f"\n--- {name} ---")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} "
              f"| Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"]))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall", "f1_score", "cv_mean_accuracy", "cv_std_accuracy"]
    ].sort_values("f1_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)

    return results_df, fitted_pipelines, X_train, X_test, y_train, y_test, preprocessor


def plot_model_comparison(results_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results_df))
    width = 0.2
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results_df[m], width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/11_model_comparison.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    print("Saved: reports/figures/11_model_comparison.png")


def tune_best_model(results_df, X_train, y_train, preprocessor):
    """
    Hyperparameter tuning is applied to the top-2 candidates by F1-score
    (typically tree-based / SVM models benefit most). We tune whichever of
    Random Forest / SVM / KNN scored highest, since these have the most
    impactful, well-understood hyperparameters for this dataset size.
    """
    top_model_name = results_df.iloc[0]["model"]
    print(f"\nTop model by F1-score on the held-out test set: {top_model_name}")
    print("Running GridSearchCV to tune hyperparameters...")

    param_grids = {
        "Random Forest": {
            "classifier": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 8, 12, 16],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "Decision Tree": {
            "classifier": [DecisionTreeClassifier(random_state=RANDOM_STATE)],
            "classifier__max_depth": [4, 8, 12, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__criterion": ["gini", "entropy"],
        },
        "KNN": {
            "classifier": [KNeighborsClassifier()],
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
        },
        "SVM": {
            "classifier": [SVC(probability=True, random_state=RANDOM_STATE)],
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"],
        },
        "Logistic Regression": {
            "classifier": [LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)],
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
    }

    grid = param_grids.get(top_model_name, param_grids["Random Forest"])
    base_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", grid["classifier"][0])])
    search_space = {k: v for k, v in grid.items() if k != "classifier"}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(base_pipe, param_grid=search_space, cv=skf, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Best parameters found: {search.best_params_}")
    print(f"Best CV F1 (macro) score: {search.best_score_:.4f}")

    return search.best_estimator_, top_model_name, search.best_params_, search.best_score_


def feature_importance_from_pipeline(best_pipeline, top_model_name, X_test=None, y_test=None):
    """
    Extract feature importance.
    - Tree-based models: use native feature_importances_.
    - Other models (SVM, Logistic Regression, KNN): fall back to permutation
      importance computed on the held-out test set, which works for any
      fitted estimator by measuring the drop in performance when a feature's
      values are shuffled.
    """
    numeric, categorical, binary = get_feature_columns()
    try:
        preprocessor = best_pipeline.named_steps["preprocessor"]
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(ohe.get_feature_names_out(categorical))
        all_feature_names = numeric + cat_feature_names + binary

        clf = best_pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            fi_df = pd.DataFrame({
                "feature": all_feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            fi_df["method"] = "native_feature_importances"
            return fi_df

        # Fallback: permutation importance on the ORIGINAL (pre-encoding) input
        # columns, computed through the full pipeline. This works for any model.
        from sklearn.inspection import permutation_importance
        print(f"{top_model_name} has no native feature_importances_. "
              f"Computing permutation importance instead...")
        result = permutation_importance(
            best_pipeline, X_test, y_test, n_repeats=10,
            random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1
        )
        input_cols = list(X_test.columns)
        fi_df = pd.DataFrame({
            "feature": input_cols,
            "importance": result.importances_mean
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        fi_df["method"] = "permutation_importance"
        return fi_df
    except Exception as e:
        print(f"Could not compute feature importance: {e}")
        return None


def main():
    results_df, fitted_pipelines, X_train, X_test, y_train, y_test, preprocessor = train_and_compare()
    plot_model_comparison(results_df)

    best_pipeline, top_model_name, best_params, best_cv_score = tune_best_model(
        results_df, X_train, y_train, preprocessor
    )

    # Final evaluation on the held-out test set
    y_pred = best_pipeline.predict(X_test)
    final_metrics = evaluate_model(y_test, y_pred)
    print("\n" + "=" * 70)
    print(f"FINAL TUNED MODEL: {top_model_name}")
    print("=" * 70)
    print(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Test Precision (macro): {final_metrics['precision']:.4f}")
    print(f"Test Recall (macro): {final_metrics['recall']:.4f}")
    print(f"Test F1 (macro): {final_metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"]))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance (if applicable)
    fi_df = feature_importance_from_pipeline(best_pipeline, top_model_name, X_test, y_test)
    if fi_df is not None:
        fi_df.to_csv("reports/feature_importance.csv", index=False)
        print("\nTop 10 important features:")
        print(fi_df.head(10).to_string(index=False))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        top_fi = fi_df.head(12)
        ax.barh(top_fi["feature"][::-1], top_fi["importance"][::-1], color="teal")
        ax.set_title(f"Feature Importance ({top_model_name})")
        ax.set_xlabel("Importance")
        os.makedirs("reports/figures", exist_ok=True)
        fig.savefig("reports/figures/12_feature_importance.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        print("Saved: reports/figures/12_feature_importance.png")

    # Save the final pipeline (preprocessing + model together)
    model_path = os.path.join(MODEL_DIR, "burning_feet_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"\nFinal pipeline saved to: {model_path}")

    # Save metadata (used later by the Streamlit app / predict.py)
    numeric, categorical, binary = get_feature_columns()
    metadata = {
        "model_name": top_model_name,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "best_cv_f1_macro": best_cv_score,
        "test_metrics": final_metrics,
        "feature_columns": {
            "numeric": numeric,
            "categorical": categorical,
            "binary": binary,
        },
        "target_classes": sorted(y_train.unique().tolist()),
        "disclaimer": (
            "This model is trained on a SYNTHETIC/DEMONSTRATION dataset for academic "
            "purposes only. It does NOT provide a medical diagnosis."
        ),
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {os.path.join(MODEL_DIR, 'model_metadata.json')}")


if __name__ == "__main__":
    main()
