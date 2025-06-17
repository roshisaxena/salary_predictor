import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction App")
st.subheader("Predict your salary based on experience")
st.write("Select your years of experience to see the estimated salary.")

# 1. Dropdown for Years of Experience (0 to 20 in steps of 0.5)
years_options = [round(x * 0.5, 1) for x in range(0, 41)]
years_exp = st.selectbox("Years of Experience:", years_options)

# 2. Predict Salary
if st.button("Predict Salary"):
    input_data = np.array([[years_exp]])
    input_scaled = scaler.transform(input_data)
    predicted_salary = model.predict(input_scaled)
    st.success(f"Estimated Salary for {years_exp} years: ₹ {predicted_salary[0][0]:,.2f}")
