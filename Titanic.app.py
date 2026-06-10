import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .title-section {
        background: linear-gradient(135deg, #1a1f35 0%, #0e1117 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .title-section h1 {
        color: #e8eaf6;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .title-section p {
        color: #7986cb;
        font-size: 1rem;
        margin: 0;
    }

    .result-survived {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        border: 1px solid #4caf50;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #c8e6c9;
    }
    .result-perished {
        background: linear-gradient(135deg, #7f0000 0%, #b71c1c 100%);
        border: 1px solid #ef5350;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #ffcdd2;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-sub {
        font-size: 0.95rem;
        opacity: 0.85;
    }

    .metric-box {
        background: #1a1f35;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-box .label {
        color: #7986cb;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-box .value {
        color: #e8eaf6;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .section-header {
        color: #7986cb;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2d3561;
    }

    div[data-testid="stSelectbox"] > label,
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stSlider"] > label {
        color: #9fa8da !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    .stMultiSelect > label { color: #9fa8da !important; font-size: 0.85rem !important; }
    div[data-testid="stCheckbox"] > label { color: #c5cae9 !important; }
</style>
""", unsafe_allow_html=True)


# ── Train all models (cached) ──────────────────────────────────────────────────
@st.cache_resource
def train_models():
    df = sns.load_dataset('titanic')

    columns_to_drop = ["deck", "embark_town", "alive", "class", "who", "adult_male"]
    df.drop(columns=columns_to_drop, inplace=True)

    df['age'].fillna(df['age'].mean(), inplace=True)
    df.dropna(subset=['embarked'], inplace=True)

    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['embarked'] = le.fit_transform(df['embarked'])

    X = df.drop("survived", axis=1)
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train all 5 models
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)

    # Accuracies
    accuracies = {
        "Logistic Regression": accuracy_score(y_test, lr.predict(X_test)),
        "KNN":                  accuracy_score(y_test, knn.predict(X_test_scaled)),
        "Naive Bayes":          accuracy_score(y_test, gnb.predict(X_test)),
        "Decision Tree":        accuracy_score(y_test, dt.predict(X_test_scaled)),
        "SVM":                  accuracy_score(y_test, svm.predict(X_test_scaled)),
    }

    return {
        "lr": lr, "knn": knn, "gnb": gnb, "dt": dt, "svm": svm,
        "scaler": scaler, "accuracies": accuracies,
        "X_test": X_test, "X_test_scaled": X_test_scaled, "y_test": y_test
    }


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-section">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p>Compare 5 ML classifiers — Logistic Regression · KNN · Naive Bayes · Decision Tree · SVM</p>
</div>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
with st.spinner("Training models on Titanic dataset..."):
    bundle = train_models()

lr       = bundle["lr"]
knn      = bundle["knn"]
gnb      = bundle["gnb"]
dt       = bundle["dt"]
svm      = bundle["svm"]
scaler   = bundle["scaler"]
accs     = bundle["accuracies"]
X_test   = bundle["X_test"]
Xs_test  = bundle["X_test_scaled"]
y_test   = bundle["y_test"]


# ── Layout: left = input form, right = results ─────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="section-header">Passenger Details</div>', unsafe_allow_html=True)

    pclass = st.selectbox("Passenger class", options=[1, 2, 3],
                          format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} class")

    sex_label = st.selectbox("Sex", options=["Male", "Female"])
    sex = 0 if sex_label == "Male" else 1

    age = st.slider("Age", min_value=1, max_value=80, value=28)

    c1, c2 = st.columns(2)
    with c1:
        sibsp = st.number_input("Siblings / spouses", min_value=0, max_value=8, value=0)
    with c2:
        parch = st.number_input("Parents / children", min_value=0, max_value=6, value=0)

    fare = st.number_input("Fare (£)", min_value=0.0, max_value=512.0, value=32.0, step=1.0)

    embarked_label = st.selectbox("Port of embarkation",
                                   options=["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    embarked = ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"].index(embarked_label)

    alone = 1 if (sibsp == 0 and parch == 0) else 0
    st.caption(f"Travelling alone: **{'Yes' if alone else 'No'}** (auto-detected)")

    st.markdown('<div class="section-header" style="margin-top:1.2rem;">Classifiers</div>', unsafe_allow_html=True)
    selected_models = st.multiselect(
        "Select models to run",
        options=["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "SVM"],
        default=["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "SVM"]
    )

    predict_btn = st.button("🔮 Predict Survival", type="primary", use_container_width=True)


# ── Right column: results ──────────────────────────────────────────────────────
with right_col:
    if predict_btn:
        if not selected_models:
            st.warning("Select at least one classifier.")
        else:
            input_raw   = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, alone]])
            input_scaled = scaler.transform(input_raw)

            model_map = {
                "Logistic Regression": (lr,  input_raw,    False),
                "KNN":                 (knn, input_scaled, True),
                "Naive Bayes":         (gnb, input_raw,    False),
                "Decision Tree":       (dt,  input_scaled, True),
                "SVM":                 (svm, input_scaled, True),
            }

            results = []
            for name in selected_models:
                model, inp, _ = model_map[name]
                pred = model.predict(inp)[0]
                prob = model.predict_proba(inp)[0][1]
                results.append({"model": name, "survived": pred, "prob": prob})

            # Consensus
            survived_votes = sum(r["survived"] for r in results)
            avg_prob = np.mean([r["prob"] for r in results])
            consensus = survived_votes > len(results) / 2

            if consensus:
                st.markdown(f"""
                <div class="result-survived">
                    <div class="result-title">✅ Survived</div>
                    <div class="result-sub">Majority of models predict survival</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-perished">
                    <div class="result-title">💀 Perished</div>
                    <div class="result-sub">Majority of models predict death</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="label">Avg. Prob</div><div class="value">{avg_prob*100:.0f}%</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="label">Survived votes</div><div class="value">{survived_votes}/{len(results)}</div></div>', unsafe_allow_html=True)
            with m3:
                best = max(results, key=lambda r: r["prob"] if r["survived"] else 0)
                st.markdown(f'<div class="metric-box"><div class="label">Models run</div><div class="value">{len(results)}</div></div>', unsafe_allow_html=True)
            with m4:
                avg_acc = np.mean([accs[r["model"]] for r in results])
                st.markdown(f'<div class="metric-box"><div class="label">Avg. Accuracy</div><div class="value">{avg_acc*100:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar chart
            st.markdown('<div class="section-header">Survival Probability by Classifier</div>', unsafe_allow_html=True)
            df_res = pd.DataFrame(results)
            colors = ["#4caf50" if s else "#ef5350" for s in df_res["survived"]]

            fig = go.Figure(go.Bar(
                x=df_res["model"],
                y=(df_res["prob"] * 100).round(1),
                marker_color=colors,
                text=(df_res["prob"] * 100).round(1).astype(str) + "%",
                textposition="outside",
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="#7986cb",
                          annotation_text="50% threshold", annotation_position="bottom right")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,31,53,0.6)",
                font=dict(color="#c5cae9", size=12),
                xaxis=dict(gridcolor="#2d3561", title=None),
                yaxis=dict(gridcolor="#2d3561", title="Survival probability (%)", range=[0, 115]),
                showlegend=False,
                margin=dict(t=20, b=10, l=10, r=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Model accuracy comparison (always visible) ──────────────────────────
    st.markdown('<div class="section-header">Model Accuracies (test set)</div>', unsafe_allow_html=True)

    fig2 = go.Figure(go.Bar(
        x=list(accs.keys()),
        y=[v * 100 for v in accs.values()],
        marker_color=["#7986cb"] * 5,
        text=[f"{v*100:.1f}%" for v in accs.values()],
        textposition="outside",
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,53,0.6)",
        font=dict(color="#c5cae9", size=12),
        xaxis=dict(gridcolor="#2d3561", title=None),
        yaxis=dict(gridcolor="#2d3561", title="Accuracy (%)", range=[0, 115]),
        showlegend=False,
        margin=dict(t=20, b=10, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Confusion matrix for one model ─────────────────────────────────────
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_model = st.selectbox("Model", options=list(accs.keys()), label_visibility="collapsed")

    cm_map = {
        "Logistic Regression": (lr,  X_test,  False),
        "KNN":                 (knn, Xs_test, True),
        "Naive Bayes":         (gnb, X_test,  False),
        "Decision Tree":       (dt,  Xs_test, True),
        "SVM":                 (svm, Xs_test, True),
    }
    model_cm, inp_cm, _ = cm_map[cm_model]
    cm = confusion_matrix(y_test, model_cm.predict(inp_cm))

    fig3 = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Perished", "Survived"],
        y=["Perished", "Survived"],
        text_auto=True,
        color_continuous_scale=[[0, "#1a1f35"], [1, "#3949ab"]],
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c5cae9", size=13),
        margin=dict(t=10, b=10, l=10, r=10),
        height=230,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)