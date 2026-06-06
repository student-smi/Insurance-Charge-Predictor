import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv('heart.csv')
    chole_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
    df['Cholesterol'] = df['Cholesterol'].replace(0, chole_mean).round(2)
    rest_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
    df['RestingBP'] = df['RestingBP'].replace(0, rest_mean).round(2)
    df_enc = pd.get_dummies(df, drop_first=True).astype(int)
    Num = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    df_enc[Num] = scaler.fit_transform(df_enc[Num])
    X = df_enc.drop(columns='HeartDisease', axis=1)
    y = df_enc['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, scaler, X.columns.tolist()

st.markdown("""
<style>
    .risk-high { background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px; }
    .risk-low  { background: #dcfce7; border-left: 4px solid #22c55e; padding: 1rem; border-radius: 8px; }
    .risk-med  { background: #fef9c3; border-left: 4px solid #eab308; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🫀 Heart Disease Risk Predictor")
st.caption("Linear Regression based prediction — enter patient details below")

model, scaler, feature_cols = train_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 90, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 7.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

if submitted:
    input_dict = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_bs == "Yes" else 0,
        'MaxHR': max_hr, 'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == "Male" else 0,
        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
        'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == "Yes" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Up" else 0,
    }
    
    input_df = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_cols}])
    Num = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[Num] = scaler.transform(input_df[Num])
    
    pred = model.predict(input_df)[0]
    risk_pct = max(0, min(100, pred * 100))
    
    st.divider()
    st.subheader("Prediction Result")
    
    col_r, col_g = st.columns([1, 2])
    with col_r:
        st.metric("Risk Score", f"{risk_pct:.1f}%")
    with col_g:
        st.progress(risk_pct / 100)
    
    if risk_pct >= 60:
        st.markdown(f'<div class="risk-high">⚠️ <strong>High Risk</strong> — Score: {risk_pct:.1f}%<br>Consult a cardiologist immediately.</div>', unsafe_allow_html=True)
    elif risk_pct >= 35:
        st.markdown(f'<div class="risk-med">⚡ <strong>Moderate Risk</strong> — Score: {risk_pct:.1f}%<br>Schedule a medical check-up.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low">✅ <strong>Low Risk</strong> — Score: {risk_pct:.1f}%<br>Keep maintaining a healthy lifestyle!</div>', unsafe_allow_html=True)

st.divider()
st.caption("Model: Linear Regression | Dataset: UCI Heart Disease | For educational purposes only")