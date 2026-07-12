from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st
from huggingface_hub import InferenceClient

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Patient Health Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------

@st.cache_resource
def load_models():

    model = joblib.load("models/diabetes_xgb.pkl")

    gender_encoder = joblib.load("models/gender_encoder.pkl")

    smoking_encoder = joblib.load("models/smoking_encoder.pkl")

    explainer = shap.TreeExplainer(model)

    return (
        model,
        gender_encoder,
        smoking_encoder,
        explainer
    )

(
    model,
    gender_encoder,
    smoking_encoder,
    explainer
) = load_models()

# ----------------------------------------------------
# DATABASE
# ----------------------------------------------------

conn = sqlite3.connect(
    "patient_reports.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS reports(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        risk_level TEXT,

        probability REAL,

        report TEXT

    )
"""
)

conn.commit()

# ----------------------------------------------------
# SAVE REPORT
# ----------------------------------------------------

def save_report(
    risk_level,
    probability,
    report
):

    cursor.execute(
        """
        INSERT INTO reports
        (
            risk_level,
            probability,
            report
        )
        VALUES
        (?, ?, ?)
        """,
        (
            risk_level,
            probability,
            report
        )
    )

    conn.commit()

# ----------------------------------------------------
# HUGGING FACE CLIENT
# ----------------------------------------------------

from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

# ----------------------------------------------------
# PREDICT RISK
# ----------------------------------------------------

def predict_risk(patient_data):

    data = patient_data.copy()

    data["gender"] = gender_encoder.transform(
        [data["gender"]]
    )[0]

    data["smoking_history"] = smoking_encoder.transform(
        [data["smoking_history"]]
    )[0]

    df = pd.DataFrame([data])

    probability = float(
        model.predict_proba(df)[0][1]
    )

    if probability >= 0.70:
        risk_level = "High"

    elif probability >= 0.40:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return probability, risk_level, df


# ----------------------------------------------------
# SHAP EXPLANATION
# ----------------------------------------------------

def explain_prediction(df):

    shap_values = explainer.shap_values(df)

    importance = {}

    for i, feature in enumerate(df.columns):

        importance[feature] = float(
            shap_values[0][i]
        )

    sorted_features = sorted(

        importance.items(),

        key=lambda x: abs(x[1]),

        reverse=True

    )

    return sorted_features[:5]


# ----------------------------------------------------
# AI REPORT
# ----------------------------------------------------

def generate_report(
    risk_level,
    probability,
    factors
):

    factor_text = ""

    for name, value in factors:
        factor_text += f"- {name}: {value:.3f}\n"

    prompt = f"""
You are an experienced healthcare assistant.

Patient Risk Level:
{risk_level}

Disease Probability:
{probability*100:.2f}%

Top Risk Factors:

{factor_text}

Generate a professional report with the following sections:

1. Risk Summary
2. Explanation
3. Lifestyle Recommendations
4. Diet Suggestions
5. Doctor Consultation Advice

Limit the response to about 180 words.
"""

    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=350
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Hugging Face Error:\n\n{e}"
# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------

st.title("🩺 AI-Powered Patient Health Risk Assessment")

st.markdown(
"""
This application predicts diabetes risk using an XGBoost model,
explains the prediction using SHAP Explainable AI,
and generates an AI-powered health report using Hugging Face Llama.
"""
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1]
    )

with col2:

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "No Info"
        ]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5
    )

    glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=400,
        value=100
    )

st.markdown("---")
# ----------------------------------------------------
# ASSESS HEALTH RISK
# ----------------------------------------------------

if st.button("🔍 Assess Health Risk"):

    patient = {

        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose

    }

    with st.spinner("Analyzing patient health..."):

        probability, risk_level, df = predict_risk(patient)

        important_factors = explain_prediction(df)

        report = generate_report(
            risk_level,
            probability,
            important_factors
        )

        save_report(
            risk_level,
            probability,
            report
        )

        st.session_state["probability"] = probability
        st.session_state["risk_level"] = risk_level
        st.session_state["important_factors"] = important_factors
        st.session_state["report"] = report

if "important_factors" in st.session_state:

    risk_level = st.session_state["risk_level"]
    probability = st.session_state["probability"]
    important_factors = st.session_state["important_factors"]
    report = st.session_state["report"]


    st.success("Assessment Completed Successfully!")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:

        if risk_level == "Low":
           st.success("🟢 LOW RISK")

        elif risk_level == "Medium":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

        st.metric(
            "Disease Probability",
            f"{probability*100:.2f}%"
        )

    with col2:

        st.subheader("🩺 Clinical Assessment")

        if risk_level == "Low":
            st.success(
                """
                **Excellent Health Status**

                ✔ Very low probability of diabetes.

                ✔ Blood sugar indicators are within a healthy range.

                ✔ Continue maintaining a healthy lifestyle with regular exercise,
                balanced nutrition, and annual health check-ups.
                """
            )

        elif risk_level == "Medium":
            st.warning(
                """
                **Moderate Diabetes Risk**

                • Some clinical indicators require attention.

                • Improve diet and physical activity.

                • Monitor blood glucose regularly.

                • Schedule a routine consultation with your physician.
                """
            )

        else:
            st.error(
                """
                **High Diabetes Risk**

                • Multiple risk factors strongly indicate diabetes.

                • Seek medical consultation as soon as possible.

                • Additional diagnostic tests are recommended.

                • Lifestyle modifications should begin immediately.
                """
            )

    st.markdown("---")

    import plotly.express as px

    st.subheader("📊 Explainable AI - Top Risk Factors")

    factor_df = pd.DataFrame(
        important_factors,
        columns=["Feature", "SHAP Value"]
    )
    feature_names = {
        "HbA1c_level": "HbA1c Level",
        "blood_glucose_level": "Blood Glucose",
        "heart_disease": "Heart Disease",
        "hypertension": "Hypertension",
        "bmi": "Body Mass Index (BMI)",
        "age": "Age",
        "gender": "Gender",
        "smoking_history": "Smoking History"
    }

    factor_df["Feature"] = factor_df["Feature"].replace(feature_names)

    # Sort by actual SHAP values
    factor_df = factor_df.reindex(
        factor_df["SHAP Value"].abs().sort_values(ascending=False).index
    )

    # Create contribution labels
    factor_df["Contribution"] = factor_df["SHAP Value"].apply(
        lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
    )

    fig = px.bar(
        factor_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Contribution",
        color_discrete_map={
            "Increases Risk": "#E74C3C",
            "Decreases Risk": "#2ECC71"
        },
        text="SHAP Value",
        title="Explainable AI - Feature Contribution"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        xaxis_title="SHAP Value",
        yaxis_title="Clinical Feature",
        legend_title="Effect on Risk",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True

    )
    st.info("""
    ### 🧠 How to Interpret This Graph

    🟢 **Green bars** indicate factors that reduced the predicted diabetes risk.

    🔴 **Red bars** indicate factors that increased the predicted diabetes risk.

    The longer the bar, the greater the influence of that feature on the AI model's decision.

    This explanation is generated using **SHAP (SHapley Additive exPlanations)**, making the prediction transparent and clinically interpretable.
    """)

    st.markdown("---")

    st.subheader("AI Generated Health Report")

    st.write(report)