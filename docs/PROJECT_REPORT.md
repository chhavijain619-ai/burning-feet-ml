# B.Tech Project Report
## Machine Learning Based Burning Feet Syndrome Symptom Assessment

---

## Chapter 1 — Introduction

Burning Feet Syndrome is a general term used for a cluster of foot sensations — burning,
tingling, numbness, and pain — that are often discussed in relation to conditions such as
diabetes, vitamin deficiencies, thyroid disorders, alcohol-related nerve damage, and peripheral
neuropathy. This project explores how a supervised machine learning classification pipeline can
be built, trained, evaluated, and deployed to produce a **preliminary, non-diagnostic assessment
category** from a set of self-reported symptoms and health-history factors.

The project is intended purely as an academic exercise in applied machine learning: dataset
design, preprocessing, model comparison, evaluation, explainability, and deployment — not as a
medical tool.

## Chapter 2 — Problem Statement

Given a set of symptom indicators (e.g., burning sensation, tingling, numbness), duration and
severity information, and relevant health-history factors (e.g., diabetes, vitamin deficiency),
classify the input into one of three preliminary assessment categories: **Low**, **Moderate**, or
**High**. This is framed as a multi-class supervised classification problem.

## Chapter 3 — Objectives

1. Design and generate a realistic, internally-consistent synthetic dataset suitable for this
   classification task.
2. Build a robust, leak-free preprocessing pipeline (imputation, encoding, scaling).
3. Perform thorough exploratory data analysis (EDA) with meaningful visualizations.
4. Engineer informative derived features without leaking the target.
5. Train and fairly compare five classification algorithms.
6. Select the best model based on measured validation performance, not assumption.
7. Tune the best model's hyperparameters using GridSearchCV.
8. Package the final pipeline for reuse and build a Streamlit interface around it.
9. Provide model explainability (feature importance / permutation importance).
10. Document, test, and deploy the system.

## Chapter 4 — Literature / Background

Foot sensations such as burning, tingling and numbness are frequently discussed in general health
literature in connection with peripheral neuropathy, diabetes-related nerve damage, vitamin B12
deficiency, thyroid dysfunction, and chronic alcohol use. Machine learning classification
techniques — Logistic Regression, Decision Trees, ensemble methods like Random Forests,
instance-based methods like KNN, and margin-based methods like SVM — are standard, well-studied
approaches for structured/tabular health-style data of this kind. This project applies these
standard techniques to a synthetic dataset built for teaching purposes.

## Chapter 5 — Proposed Methodology

The project follows a standard applied-ML workflow: data generation/loading → cleaning → EDA →
visualization → feature engineering → encoding/scaling (via a `ColumnTransformer` inside a
`Pipeline`, to avoid data leakage) → stratified train-test split → training multiple candidate
models → cross-validated comparison → hyperparameter tuning of the best candidate →
final evaluation → model persistence (`joblib`) → deployment through a Streamlit UI.

## Chapter 6 — Dataset

A synthetic/demonstration dataset of ~1500 records was generated (`generate_dataset.py`) using
randomized but internally consistent rules — e.g., diabetes history and peripheral neuropathy
history increase the probability of burning/tingling/numbness symptoms and higher severity
scores, which in turn shift the probability of the target class toward `High`. The dataset
includes demographic fields (age, gender), eight binary symptom indicators, duration and severity
fields, five binary health-history indicators, a physical-activity category, and the three-class
target `assessment`. A small number of missing values and duplicate rows were deliberately
introduced to allow realistic data-cleaning practice.

## Chapter 7 — Data Preprocessing

Preprocessing steps included: removing duplicate rows, median-imputing missing numeric values,
mode-imputing missing categorical values, clipping out-of-range values, and building a
`ColumnTransformer` that applies `StandardScaler` to numeric features and `OneHotEncoder` to
categorical features, while passing already-binary indicator columns through. This entire
transformer is embedded inside a single `sklearn.Pipeline` together with the classifier, so that
scaling/encoding statistics are always fit only on training data — preventing data leakage.

## Chapter 8 — Exploratory Data Analysis

EDA included dataset shape, dtypes, missing-value counts, duplicate counts, and descriptive
statistics (mean, median, std, min, max). Ten visualizations were produced: class distribution,
age distribution, gender distribution, three individual symptom distributions (burning, tingling,
numbness), two symptom-vs-target comparisons (diabetes and vitamin deficiency vs. assessment),
severity distribution, and a correlation heatmap across numeric features.

## Chapter 9 — Machine Learning Algorithms

Five classification algorithms were trained and compared:
1. **Logistic Regression** — a linear, probabilistic baseline classifier.
2. **Decision Tree** — a simple, interpretable rule-based model.
3. **Random Forest** — an ensemble of decision trees that reduces overfitting via bagging.
4. **K-Nearest Neighbors (KNN)** — an instance-based, non-parametric classifier.
5. **Support Vector Machine (SVM)** — a margin-maximizing classifier, here with an RBF kernel.

## Chapter 10 — Model Training and Evaluation

Each model was wrapped in the same preprocessing pipeline and evaluated using **stratified 5-fold
cross-validation** on the training split, then re-fit on the full training split and evaluated on
a held-out 20% test split. Metrics recorded: Accuracy, macro-averaged Precision, Recall, and
F1-score, plus confusion matrices and full classification reports.

## Chapter 11 — Results

Across the run documented in this project, the models were ranked as follows (macro F1-score,
test set): SVM ≈ Logistic Regression > Random Forest > Decision Tree > KNN. The tuned SVM
(hyperparameters selected via `GridSearchCV` over `C`, `kernel`, `gamma`) was chosen as the final
model with approximately **87% test accuracy and macro F1-score**. Exact numbers are stored in
`reports/model_comparison.csv` and will vary slightly with different random seeds or dataset
regeneration. *These figures describe performance on synthetic data only.*

## Chapter 12 — Streamlit Application

A Streamlit application (`app.py`) was built to demonstrate the trained pipeline. It presents an
input form for symptoms and health history, a prominent academic/medical disclaimer, a
predicted assessment category with class-probability visualization, a summary of entered
symptoms/health factors, and a model-explainability section showing the top features driving
predictions (via permutation importance, since the winning SVM model has no native
`feature_importances_`).

## Chapter 13 — Testing

A `pytest` suite (`tests/test_prediction.py`) validates: model loading, predictions for valid
input, probability outputs summing to ~1, minimum/maximum age edge cases, high-risk and low-risk
symptom combinations, correct error handling on missing required fields, graceful handling of
unseen categorical values, and correctness of the engineered `symptom_score`,
`risk_factor_count`, and `duration_category` features.

## Chapter 14 — Limitations

- Entirely synthetic training data — no real clinical validation.
- Simplified feature set relative to real clinical assessment tools.
- Class boundaries defined by an arbitrary synthetic composite score, not medical consensus.
- No external validation against real patient outcomes.

## Chapter 15 — Future Scope

- Integrate properly consented, de-identified real-world data under clinical supervision.
- Add validated neuropathy/symptom questionnaires as richer input features.
- Experiment with gradient-boosted trees (XGBoost/LightGBM) and simple neural networks.
- Add a REST API layer and/or user history tracking with appropriate privacy safeguards.

## Chapter 16 — Conclusion

This project demonstrates a complete, reproducible applied machine-learning workflow — from
synthetic dataset design through EDA, preprocessing, multi-model comparison, hyperparameter
tuning, explainability, testing, and deployment — culminating in a simple, disclaimer-forward
Streamlit application. It fulfills its purpose as an academic B.Tech ML project while being
explicit throughout that it does not constitute a medical diagnostic tool.
