"""
Insurance Charges Prediction Model
====================================
Converted from Untitled1.ipynb

Dataset : insurance.csv
Target   : charges (medical insurance cost)
Model    : Linear Regression (with StandardScaler preprocessing)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  INSURANCE CHARGES PREDICTION MODEL")
print("=" * 60)

df = pd.read_csv('insurance.csv')

print("\n[1] Dataset Preview (first 5 rows):")
print(df.head())

print(f"\n[2] Shape: {df.shape}")

print("\n[3] Data Info:")
df.info()

print("\n[4] Statistical Summary:")
print(df.describe())

print("\n[5] Missing Values:")
print(df.isnull().sum())

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Distribution of charges
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['charges'], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribution of Insurance Charges')
axes[0].set_xlabel('Charges ($)')

sns.boxplot(data=df, x='smoker', y='charges', ax=axes[1], palette='Set2')
axes[1].set_title('Charges by Smoker Status')
plt.tight_layout()
plt.savefig('eda_charges.png', dpi=100)
plt.show()
print("  → Saved: eda_charges.png")

# Correlation heatmap (numeric only)
fig, ax = plt.subplots(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap (Numeric Features)')
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=100)
plt.show()
print("  → Saved: eda_correlation.png")

# Charges vs Age, coloured by smoker
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker', palette={'yes': '#e74c3c', 'no': '#2ecc71'}, alpha=0.7, ax=ax)
ax.set_title('Charges vs Age (coloured by Smoker Status)')
plt.tight_layout()
plt.savefig('eda_age_vs_charges.png', dpi=100)
plt.show()
print("  → Saved: eda_age_vs_charges.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURE ENGINEERING & PREPROCESSING")
print("=" * 60)

df_model = df.copy()

# 3a. Encode binary categorical columns
df_model['sex']    = df_model['sex'].map({'female': 1, 'male': 0})          # is_female
df_model['smoker'] = df_model['smoker'].map({'yes': 1, 'no': 0})            # is_smoker
df_model.rename(columns={'sex': 'is_female', 'smoker': 'is_smoker'}, inplace=True)

# 3b. One-hot encode 'region' (drop first to avoid multicollinearity)
df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)

# 3c. Create BMI category feature
df_model['bmi_category'] = pd.cut(
    df_model['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# One-hot encode BMI category
df_model = pd.get_dummies(df_model, columns=['bmi_category'], drop_first=True)

# 3d. Convert boolean dummy columns to int
bool_cols = df_model.select_dtypes(include='bool').columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

print("\nCleaned DataFrame columns:")
print(list(df_model.columns))
print(f"\nShape after feature engineering: {df_model.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFINE FEATURES & TARGET
# ─────────────────────────────────────────────────────────────────────────────
target = 'charges'
feature_cols = [c for c in df_model.columns if c != target]

X = df_model[feature_cols].copy()
y = df_model[target].copy()

# 3e. Scale continuous features
scaler = StandardScaler()
scale_cols = ['age', 'bmi', 'children']
X[scale_cols] = scaler.fit_transform(X[scale_cols])

print(f"\nFeatures used ({len(feature_cols)}):")
print(feature_cols)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

model = LinearRegression()
model.fit(X_train, y_train)
print("  Training complete ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL EVALUATION")
print("=" * 60)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n  Train R²  : {r2_train:.4f}")
print(f"  Test  R²  : {r2_test:.4f}")
print(f"  Test  MAE : ${mae_test:,.2f}")
print(f"  Test  RMSE: ${rmse_test:,.2f}")

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature'    : feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n  Top feature coefficients:")
print(coef_df.to_string(index=False))

# Actual vs Predicted plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_test, alpha=0.5, color='royalblue', edgecolors='white', linewidth=0.3)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Charges ($)')
ax.set_ylabel('Predicted Charges ($)')
ax.set_title(f'Actual vs Predicted (Test R² = {r2_test:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig('model_actual_vs_predicted.png', dpi=100)
plt.show()
print("\n  → Saved: model_actual_vs_predicted.png")

# Residuals plot
residuals = y_test - y_pred_test
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred_test, residuals, alpha=0.5, color='coral', edgecolors='white', linewidth=0.3)
ax.axhline(0, color='black', lw=1.5, linestyle='--')
ax.set_xlabel('Predicted Charges ($)')
ax.set_ylabel('Residuals ($)')
ax.set_title('Residual Plot')
plt.tight_layout()
plt.savefig('model_residuals.png', dpi=100)
plt.show()
print("  → Saved: model_residuals.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE MODEL & SCALER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SAVING MODEL & SCALER")
print("=" * 60)

joblib.dump(model,  'insurance_model.pkl')
joblib.dump(scaler, 'insurance_scaler.pkl')
print("  → Saved: insurance_model.pkl")
print("  → Saved: insurance_scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 9. QUICK PREDICTION DEMO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREDICTION DEMO (Sample New Person)")
print("=" * 60)

# Load saved model & scaler
loaded_model  = joblib.load('insurance_model.pkl')
loaded_scaler = joblib.load('insurance_scaler.pkl')

# Sample: 35-year-old male smoker, BMI 28, 1 child, northwest region
sample_raw = pd.DataFrame([{
    'age'                     : 35,
    'is_female'               : 0,
    'bmi'                     : 28.0,
    'children'                : 1,
    'is_smoker'               : 1,
    'region_northwest'        : 1,
    'region_southeast'        : 0,
    'region_southwest'        : 0,
    'bmi_category_Normal'     : 0,
    'bmi_category_Overweight' : 1,
    'bmi_category_Obese'      : 0,
}])

# Ensure column order matches training
sample_raw = sample_raw.reindex(columns=feature_cols, fill_value=0)
sample_raw[scale_cols] = loaded_scaler.transform(sample_raw[scale_cols])

prediction = loaded_model.predict(sample_raw)[0]
print(f"\n  Input  : Age=35, Male, BMI=28, 1 child, Smoker, Northwest")
print(f"  Predicted Charges : ${prediction:,.2f}")

print("\n" + "=" * 60)
print("  DONE — Model pipeline complete!")
print("=" * 60)
