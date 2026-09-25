"""
app.py
------
Streamlit application: Burning Feet Syndrome - ML-Based Symptom Assessment

Run:
    streamlit run app.py
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_assessment, load_artifacts

st.set_page_config(
    page_title="Burning Feet Syndrome - ML Symptom Assessment",
    page_icon="🦶",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header & Disclaimer
# ---------------------------------------------------------------------------
st.title("🦶 Burning Feet Syndrome — ML-Based Symptom Assessment")
st.caption("An academic B.Tech Machine Learning project")

st.warning(
    "**Disclaimer:** This application is an academic machine-learning project and "
    "does **not** provide a medical diagnosis. The prediction is based on a model "
    "trained on a **synthetic/demonstration dataset** and should not be used as a "
    "substitute for professional medical advice. If you are experiencing real "
    "symptoms, please consult a qualified healthcare professional."
)

# Load model artifacts once (cached)
try:
    pipeline, metadata = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(str(e))
    st.stop()

with st.expander("ℹ️ About this model"):
    st.write(f"**Model used:** {metadata['model_name']}")
    st.write(f"**Test Accuracy:** {metadata['test_metrics']['accuracy']:.2%}")
    st.write(f"**Test F1-score (macro):** {metadata['test_metrics']['f1_score']:.2%}")
    st.write(metadata["disclaimer"])

st.divider()

# ---------------------------------------------------------------------------
# User Input Form
# ---------------------------------------------------------------------------
st.subheader("📋 Enter Symptom & Health Information")

with st.form("assessment_form"):
    st.markdown("### Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=110, value=40, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("### Symptoms")
    col1, col2 = st.columns(2)
    with col1:
        burning_sensation = st.checkbox("Burning sensation in feet")
        tingling = st.checkbox("Tingling sensation")
        numbness = st.checkbox("Numbness")
        foot_pain = st.checkbox("Foot pain")
    with col2:
        warmth_sensation = st.checkbox("Warmth sensation")
        redness = st.checkbox("Redness")
        swelling = st.checkbox("Swelling")
        night_time_symptoms = st.checkbox("Symptoms worse at night")

    duration_weeks = st.slider("Duration of symptoms (weeks)", 0.0, 52.0, 4.0, step=0.5)
    severity_level = st.slider("Overall severity level (1 = mild, 10 = severe)", 1, 10, 5)

    st.markdown("### Health Factors")
    col1, col2 = st.columns(2)
    with col1:
        diabetes_history = st.radio("History of diabetes?", ["No", "Yes"], horizontal=True)
        vitamin_deficiency_history = st.radio("History of vitamin deficiency?", ["No", "Yes"], horizontal=True)
        thyroid_condition = st.radio("History of thyroid condition?", ["No", "Yes"], horizontal=True)
    with col2:
        alcohol_risk_factor = st.radio("Alcohol-related risk factor?", ["No", "Yes"], horizontal=True)
        peripheral_neuropathy_history = st.radio("History of peripheral neuropathy?", ["No", "Yes"], horizontal=True)
        physical_activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

    submitted = st.form_submit_button("🔍 Assess Symptoms", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction & Result Visualization
# ---------------------------------------------------------------------------
if submitted:
    raw_input = {
        "age": age,
        "gender": gender,
        "burning_sensation": int(burning_sensation),
        "tingling": int(tingling),
        "numbness": int(numbness),
        "foot_pain": int(foot_pain),
        "warmth_sensation": int(warmth_sensation),
        "redness": int(redness),
        "swelling": int(swelling),
        "night_time_symptoms": int(night_time_symptoms),
        "duration_weeks": duration_weeks,
        "diabetes_history": 1 if diabetes_history == "Yes" else 0,
        "vitamin_deficiency_history": 1 if vitamin_deficiency_history == "Yes" else 0,
        "thyroid_condition": 1 if thyroid_condition == "Yes" else 0,
        "alcohol_risk_factor": 1 if alcohol_risk_factor == "Yes" else 0,
        "peripheral_neuropathy_history": 1 if peripheral_neuropathy_history == "Yes" else 0,
        "physical_activity": physical_activity,
        "severity_level": severity_level,
    }

    result = predict_assessment(raw_input)
    prediction = result["prediction"]
    probabilities = result["probabilities"]

    st.divider()
    st.subheader("📊 Preliminary Assessment")

    color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
    st.markdown(
        f"### The model classified the submitted symptom pattern as: "
        f":{color_map.get(prediction, 'blue')}[**{prediction}**]"
    )
    st.caption(
        "This is a preliminary ML-based assessment derived from a synthetic training "
        "dataset — it is **not** a clinical diagnosis of Burning Feet Syndrome."
    )

    if probabilities:
        st.markdown("#### Model Confidence (class probabilities)")
        proba_df = pd.DataFrame({
            "Category": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }).set_index("Category").loc[["Low", "Moderate", "High"]]
        st.bar_chart(proba_df)

    st.markdown("#### Key Symptoms Entered")
    entered_symptoms = []
    symptom_labels = {
        "burning_sensation": "Burning sensation", "tingling": "Tingling",
        "numbness": "Numbness", "foot_pain": "Foot pain",
        "warmth_sensation": "Warmth sensation", "redness": "Redness",
        "swelling": "Swelling", "night_time_symptoms": "Night-time symptoms",
    }
    for key, label in symptom_labels.items():
        if raw_input[key] == 1:
            entered_symptoms.append(label)
    if entered_symptoms:
        st.write(", ".join(entered_symptoms))
    else:
        st.write("No specific symptoms were selected.")

    health_labels = {
        "diabetes_history": "Diabetes history",
        "vitamin_deficiency_history": "Vitamin deficiency history",
        "thyroid_condition": "Thyroid condition",
        "alcohol_risk_factor": "Alcohol-related risk factor",
        "peripheral_neuropathy_history": "Peripheral neuropathy history",
    }
    entered_health = [label for key, label in health_labels.items() if raw_input[key] == 1]
    st.markdown("#### Relevant Health History Entered")
    st.write(", ".join(entered_health) if entered_health else "None selected.")

    st.info(
        "**General educational note:** Sensations such as burning, tingling and numbness "
        "in the feet are commonly discussed in relation to factors like blood sugar "
        "regulation, nerve health, and vitamin levels. This tool only reflects patterns "
        "learned from synthetic training data and cannot account for your real medical "
        "history. Please consult a healthcare professional for an actual diagnosis."
    )

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🔬 Model Explainability")
    fi_path = os.path.join("reports", "feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        st.markdown("**Important Features Used by the Model**")
        st.caption(
            "This shows which input features most influenced the MODEL'S behavior "
            "on the test data — it describes statistical patterns learned by the "
            "model, not medical causation."
        )
        top_fi = fi_df.head(10).set_index("feature")
        st.bar_chart(top_fi["importance"])
    else:
        st.write("Feature importance report not found. Run `python src/train_models.py` to generate it.")

st.divider()
st.caption(
    "Burning Feet Syndrome ML Symptom Assessment — Academic B.Tech Project | "
    "Not for medical use."
)
