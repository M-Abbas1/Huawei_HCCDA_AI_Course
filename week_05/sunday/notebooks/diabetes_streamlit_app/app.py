import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]

st.title("🩺 Diabetes Prediction")
st.write(
    "Enter the patient's measurements below. The trained logistic regression "
    "model will return a prediction."
)
st.caption("Educational demonstration only; this tool is not a medical diagnosis.")

with st.form("diabetes_form"):
    st.subheader("Patient measurements")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=3.0, value=0.5, step=0.01
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }])

    # Keep feature order exactly the same as during model training.
    input_data = input_data[bundle["feature_names"]]
    input_scaled = scaler.transform(input_data)
    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    diabetes_probability = float(probabilities[1])

    st.subheader("Prediction result")
    if prediction == 1:
        st.error("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No Diabetes")

    st.write(f"Model-estimated probability for the diabetes class: **{diabetes_probability:.1%}**")
    st.progress(max(0.0, min(1.0, diabetes_probability)))
    st.caption(
        "This probability is the model's estimate, not a confirmed clinical risk. "
        "Please consult a qualified healthcare professional for interpretation."
    )
