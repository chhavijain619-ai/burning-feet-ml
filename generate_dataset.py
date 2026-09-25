"""
generate_dataset.py
--------------------
Generates a SYNTHETIC / DEMONSTRATION dataset for the
"Machine Learning Based Burning Feet Syndrome Symptom Assessment" academic project.

IMPORTANT DISCLAIMER:
This data is entirely synthetic and generated programmatically using randomized,
rule-based relationships between features. It does NOT come from real patients
and must NOT be interpreted as real clinical / medical data. It exists purely
to let a classification ML pipeline be trained, evaluated and demonstrated for
academic (B.Tech) purposes.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_RECORDS = 1500

np.random.seed(RANDOM_SEED)


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_dataset(n=N_RECORDS):
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------- Base demographic features ----------
    age = rng.integers(18, 80, size=n)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.48, 0.52])

    # ---------- Underlying "latent risk" (not stored, only used to build realistic correlations) ----------
    # Older age + certain conditions increase latent risk of higher severity.
    diabetes_prob = clip(0.10 + (age - 18) / 400, 0.05, 0.45)
    diabetes = rng.binomial(1, diabetes_prob)

    vitamin_deficiency_prob = np.where(gender == "Female", 0.30, 0.22)
    vitamin_deficiency = rng.binomial(1, vitamin_deficiency_prob)

    thyroid_prob = np.where(gender == "Female", 0.18, 0.10)
    thyroid = rng.binomial(1, thyroid_prob)

    alcohol_risk_prob = np.where(gender == "Male", 0.25, 0.12)
    alcohol_risk = rng.binomial(1, alcohol_risk_prob)

    neuropathy_prob = clip(0.05 + diabetes * 0.25 + (age - 18) / 500, 0.03, 0.5)
    peripheral_neuropathy = rng.binomial(1, neuropathy_prob)

    physical_activity = rng.choice(["Low", "Moderate", "High"], size=n, p=[0.35, 0.45, 0.20])

    # ---------- Latent severity score (drives symptom generation; NOT stored directly) ----------
    activity_penalty = np.select(
        [physical_activity == "Low", physical_activity == "Moderate", physical_activity == "High"],
        [0.15, 0.05, -0.05],
    )

    latent_risk = (
        0.02 * (age - 18) / 61
        + 0.25 * diabetes
        + 0.18 * vitamin_deficiency
        + 0.12 * thyroid
        + 0.10 * alcohol_risk
        + 0.30 * peripheral_neuropathy
        + activity_penalty
        + rng.normal(0, 0.12, size=n)
    )
    latent_risk = clip(latent_risk, 0, 1.6)

    # ---------- Symptom features (0/1 or ordinal), probabilistically tied to latent_risk ----------
    def prob_symptom(base, weight):
        return clip(base + weight * latent_risk, 0.02, 0.97)

    burning_sensation = rng.binomial(1, prob_symptom(0.20, 0.55))
    tingling = rng.binomial(1, prob_symptom(0.18, 0.50))
    numbness = rng.binomial(1, prob_symptom(0.12, 0.55))
    foot_pain = rng.binomial(1, prob_symptom(0.20, 0.45))
    warmth_sensation = rng.binomial(1, prob_symptom(0.15, 0.40))
    redness = rng.binomial(1, prob_symptom(0.10, 0.30))
    swelling = rng.binomial(1, prob_symptom(0.10, 0.30))
    night_time_symptoms = rng.binomial(1, prob_symptom(0.15, 0.45))

    # Duration of symptoms (in weeks) - higher latent risk -> longer duration typically
    duration_weeks = clip(
        rng.gamma(shape=2.0, scale=2.0 + latent_risk * 4, size=n), 0, 52
    ).round(1)

    # Severity level (ordinal 1-10), correlated with latent_risk and symptom count
    symptom_sum = (
        burning_sensation + tingling + numbness + foot_pain
        + warmth_sensation + redness + swelling + night_time_symptoms
    )
    severity_level = clip(
        (latent_risk * 5 + symptom_sum * 0.5 + rng.normal(0, 1.0, size=n)).round(0),
        1, 10,
    ).astype(int)

    # ---------- Target variable ----------
    # Composite score combining latent risk, symptom burden, duration and severity.
    composite_score = (
        latent_risk * 3.0
        + symptom_sum * 0.35
        + (duration_weeks / 52) * 1.5
        + (severity_level / 10) * 2.0
    )

    low_thr = np.quantile(composite_score, 0.40)
    high_thr = np.quantile(composite_score, 0.75)

    assessment = np.where(
        composite_score <= low_thr, "Low",
        np.where(composite_score <= high_thr, "Moderate", "High")
    )

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "burning_sensation": burning_sensation,
        "tingling": tingling,
        "numbness": numbness,
        "foot_pain": foot_pain,
        "warmth_sensation": warmth_sensation,
        "redness": redness,
        "swelling": swelling,
        "night_time_symptoms": night_time_symptoms,
        "duration_weeks": duration_weeks,
        "diabetes_history": diabetes,
        "vitamin_deficiency_history": vitamin_deficiency,
        "thyroid_condition": thyroid,
        "alcohol_risk_factor": alcohol_risk,
        "peripheral_neuropathy_history": peripheral_neuropathy,
        "physical_activity": physical_activity,
        "severity_level": severity_level,
        "assessment": assessment,
    })

    # Introduce a small number of realistic missing values (MCAR) to practice cleaning
    for col in ["duration_weeks", "severity_level", "physical_activity"]:
        missing_idx = rng.choice(df.index, size=int(0.01 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Introduce a handful of duplicate rows to practice de-duplication
    dup_rows = df.sample(n=8, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main():
    df = generate_dataset(N_RECORDS)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "burning_feet_dataset.csv")
    df.to_csv(out_path, index=False)

    print("=" * 60)
    print("SYNTHETIC / DEMONSTRATION DATASET GENERATED")
    print("(This is NOT real clinical data)")
    print("=" * 60)
    print(f"Saved to: {out_path}")
    print(f"Total records (incl. duplicates for cleaning practice): {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nClass distribution (assessment):")
    print(df["assessment"].value_counts())
    print("\nBasic statistics (numeric columns):")
    print(df.describe().T)


if __name__ == "__main__":
    main()
