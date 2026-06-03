import streamlit as st
import joblib, numpy as np

model = joblib.load('insurance_lr_model.pkl')
# scaler bhi load karo agar save kiya ho

st.title("🏥 Insurance Charge Predictor")

age      = st.number_input("Age", 18, 100, 30)
sex      = st.selectbox("Sex", ["Male", "Female"])
bmi      = st.number_input("BMI", 10.0, 60.0, 28.0)
children = st.number_input("Children", 0, 10, 0)
smoker   = st.selectbox("Smoker?", ["No", "Yes"])
region   = st.selectbox("Region", ["Northwest","Northeast","Southwest","Southeast"])

if st.button("Predict"):
    # same preprocessing as notebook
    is_female = 1 if sex=="Female" else 0
    is_smoker = 1 if smoker=="Yes" else 0
    region_se = 1 if region=="Southeast" else 0
    age_s = (age - 39.207025) / 14.049960
    bmi_s = (bmi - 30.663397) / 6.098187
    ch_s  = (children - 1.094918) / 1.205493
    bmi_obese = 1 if bmi >= 30 else 0

    X = np.array([[age_s, is_female, bmi_s, ch_s, is_smoker, region_se, bmi_obese]])
    pred = model.predict(X)[0]
    st.success(f"💰 Estimated Charge: ₹{pred:,.2f}")