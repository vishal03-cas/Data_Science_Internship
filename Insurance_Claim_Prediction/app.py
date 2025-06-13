import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('xgboost_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("Insurance Charges Prediction")

st.markdown("""
This app predicts **Medical Insurance Charges** based on:
- Age
- BMI
- Smoking status
""")

# User inputs
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
smoker = st.selectbox("Smoker", ["No", "Yes"])

# Encode smoker
smoker_encoded = 1 if smoker == "Yes" else 0

# Prepare input for model
input_data = np.array([[smoker_encoded, age, bmi]])

# Predict
if st.button("Predict Charges"):
    predicted_charges = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Charges: **${predicted_charges:,.2f}**")
