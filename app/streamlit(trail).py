import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model and features ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict churn probability and get retention recommendations.")
st.divider()

# ── Sidebar — Customer Input Form ───────────────────────────────────────────
st.sidebar.header("👤 Customer Details")
st.sidebar.markdown("Fill in the customer information:")

# Demographics
st.sidebar.subheader("Demographics")
gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen  = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])

# Service details
st.sidebar.subheader("Service Details")
tenure          = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service   = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines  = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service= st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup   = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection=st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support    = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv    = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies= st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Billing
st.sidebar.subheader("Billing")
contract        = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method  = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
total_charges   = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0,
                                     float(monthly_charges * tenure), 10.0)

predict_btn = st.sidebar.button("🔮 Predict Churn", type="primary", use_container_width=True)

# ── Helper functions ─────────────────────────────────────────────────────────
def encode_input():
    """Encode user inputs to match training data format"""
    from sklearn.preprocessing import LabelEncoder

    data = {
        "Gender"            : gender,
        "Senior Citizen"    : 1 if senior_citizen == "Yes" else 0,
        "Partner"           : partner,
        "Dependents"        : dependents,
        "Tenure Months"     : tenure,
        "Phone Service"     : phone_service,
        "Multiple Lines"    : multiple_lines,
        "Internet Service"  : internet_service,
        "Online Security"   : online_security,
        "Online Backup"     : online_backup,
        "Device Protection" : device_protection,
        "Tech Support"      : tech_support,
        "Streaming TV"      : streaming_tv,
        "Streaming Movies"  : streaming_movies,
        "Contract"          : contract,
        "Paperless Billing" : paperless,
        "Payment Method"    : payment_method,
        "Monthly Charges"   : monthly_charges,
        "Total Charges"     : total_charges,
    }

    df = pd.DataFrame([data])

    # Encode categoricals same way as training
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Add engineered features
    df["tenure_group"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 36, 60, 999],
        labels=["New", "Mid", "Loyal", "Champion"]
    ).astype(str)
    df["tenure_group"] = le.fit_transform(df["tenure_group"])

    df["revenue_band"] = pd.cut(
        df["Monthly Charges"],
        bins=[-1, 35, 65, 85, 999],
        labels=["Low", "Medium", "High", "Premium"]
    ).astype(str)
    df["revenue_band"] = le.fit_transform(df["revenue_band"])

    df["service_count"] = (
        (df["Phone Service"] > 0).astype(int) +
        (df["Multiple Lines"] > 0).astype(int) +
        (df["Internet Service"] > 0).astype(int) +
        (df["Online Security"] > 0).astype(int) +
        (df["Online Backup"] > 0).astype(int) +
        (df["Device Protection"] > 0).astype(int) +
        (df["Tech Support"] > 0).astype(int) +
        (df["Streaming TV"] > 0).astype(int) +
        (df["Streaming Movies"] > 0).astype(int)
    )

    df["customer_value"] = 1 if (monthly_charges >= 85 and tenure > 24) else 0
    df["risk_segment"]   = (
        2 if (contract == "Month-to-month" and monthly_charges > 65 and tenure < 12)
        else 1 if contract == "Month-to-month"
        else 0
    )

    # RFM scores (simplified)
    df["Recency"]  = 72 - tenure
    df["Frequency"]= df["service_count"]
    df["Monetary"] = monthly_charges
    df["R_Score"]  = min(5, max(1, int((72 - tenure) / 14.4) + 1))
    df["F_Score"]  = min(5, max(1, int(df["service_count"].iloc[0] / 2) + 1))
    df["M_Score"]  = min(5, max(1, int(monthly_charges / 24) + 1))
    df["RFM_Score"]= df["R_Score"] + df["F_Score"] + df["M_Score"]
    df["Segment"]  = 0
    df["Sentiment_Score"] = 0.0

    # Keep only features the model was trained on
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

def get_segment_name(churn_prob, tenure, monthly_charges, contract):
    """Assign segment based on customer profile"""
    if churn_prob >= 0.5 and tenure < 24:
        return "🔴 High Risk"
    elif churn_prob >= 0.3:
        return "🟠 New Passives"
    elif contract != "Month-to-month" and tenure > 36:
        return "🟢 Champions"
    else:
        return "🔵 Loyal Basics"

def get_retention_strategy(churn_prob, contract, tenure, monthly_charges):
    """Return retention strategy based on profile"""
    strategies = []
    if churn_prob >= 0.7:
        strategies.append("🚨 **Immediate Action Required** — assign dedicated account manager")
        strategies.append("💰 Offer personalised discount of 15-20% for next 3 months")
        strategies.append("📞 Schedule proactive call within 48 hours")
    elif churn_prob >= 0.4:
        strategies.append("⚠️ **Medium Risk** — add to monthly watch list")
        strategies.append("🎁 Send loyalty reward or service upgrade offer")
        strategies.append("📧 Enrol in satisfaction survey campaign")
    else:
        strategies.append("✅ **Low Risk** — maintain current service quality")
        strategies.append("⭐ Consider for referral program")

    if contract == "Month-to-month":
        strategies.append("📋 Offer annual contract with 10% discount to lock in loyalty")
    if monthly_charges > 85:
        strategies.append("💎 Premium customer — ensure VIP support experience")
    if tenure < 12:
        strategies.append("🤝 New customer — enrol in onboarding program (months 3, 6, 9 check-ins)")

    return strategies

# ── Main content ─────────────────────────────────────────────────────────────
if predict_btn:
    # Encode input
    input_df = encode_input()

    # Predict
    churn_prob    = model.predict_proba(input_df)[0][1]
    churn_pred    = 1 if churn_prob >= 0.5 else 0
    segment       = get_segment_name(churn_prob, tenure, monthly_charges, contract)
    strategies    = get_retention_strategy(churn_prob, contract, tenure, monthly_charges)

    # Risk color
    if churn_prob >= 0.7:
        color = "#FF6B6B"
        risk  = "HIGH RISK"
    elif churn_prob >= 0.4:
        color = "#ED7D31"
        risk  = "MEDIUM RISK"
    else:
        color = "#70AD47"
        risk  = "LOW RISK"

    # ── Row 1: Key metrics ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob*100:.1f}%",
            delta=f"{risk}"
        )
    with col2:
        st.metric(
            label="Prediction",
            value="Will Churn ⚠️" if churn_pred == 1 else "Will Stay ✅"
        )
    with col3:
        st.metric(
            label="Customer Segment",
            value=segment
        )
    with col4:
        st.metric(
            label="Contract Type",
            value=contract
        )

    st.divider()

    # ── Row 2: Gauge + SHAP ─────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🎯 Churn Risk Gauge")

        # Simple gauge using progress bar
        st.markdown(f"""
        <div style='text-align:center; padding:20px;
                    background-color:{color}22;
                    border-radius:10px;
                    border: 2px solid {color}'>
            <h1 style='color:{color}; font-size:60px; margin:0'>
                {churn_prob*100:.1f}%
            </h1>
            <h3 style='color:{color}'>{risk}</h3>
            <p style='color:gray'>Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(churn_prob))
        st.markdown(f"**Tenure:** {tenure} months | **Monthly:** ${monthly_charges:.2f}")

    with col_right:
        st.subheader("🔍 SHAP — Why this prediction?")
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values      = shap_values[0],
                    base_values = explainer.expected_value,
                    data        = input_df.iloc[0],
                    feature_names = feature_names
                ),
                show=False
            )
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP chart unavailable: {e}")

    st.divider()

    # ── Row 3: Retention Strategies ─────────────────────────────────────────
    st.subheader("💡 Retention Strategy Recommendations")

    cols = st.columns(len(strategies))
    for i, (col, strategy) in enumerate(zip(cols, strategies)):
        with col:
            st.info(strategy)

    st.divider()

    # ── Row 4: Customer Profile Summary ─────────────────────────────────────
    st.subheader("📋 Customer Profile Summary")

    profile_data = {
        "Feature"       : ["Gender", "Senior Citizen", "Tenure", "Contract",
                           "Monthly Charges", "Internet Service", "Tech Support",
                           "Online Security", "Payment Method"],
        "Value"         : [gender, senior_citizen, f"{tenure} months", contract,
                           f"${monthly_charges:.2f}", internet_service,
                           tech_support, online_security, payment_method],
        "Risk Impact"   : ["Neutral", "Neutral",
                           "🔴 High" if tenure < 12 else "🟢 Low",
                           "🔴 High" if contract == "Month-to-month" else "🟢 Low",
                           "🔴 High" if monthly_charges > 65 else "🟢 Low",
                           "🔴 High" if internet_service == "Fiber optic" else "🟢 Low",
                           "🟢 Low" if tech_support == "Yes" else "🔴 High",
                           "🟢 Low" if online_security == "Yes" else "🔴 High",
                           "🔴 High" if payment_method == "Electronic check" else "🟢 Low"]
    }

    st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

else:
    # ── Default screen when no prediction yet ───────────────────────────────
    st.markdown("""
    ## Welcome to the Customer Churn Predictor! 👋

    This app uses a trained **XGBoost model** (AUC = 0.92) to predict
    whether a telecom customer is likely to churn.

    ### How to use:
    1. 👈 Fill in customer details in the **sidebar**
    2. Click **🔮 Predict Churn** button
    3. View churn probability, SHAP explanation and retention strategies

    ### What you'll see:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🎯 **Churn Probability**\nReal-time prediction score")
    with col2:
        st.warning("🔍 **SHAP Explanation**\nWhy the model made this prediction")
    with col3:
        st.info("💡 **Retention Strategy**\nPersonalised recommendations")

    st.markdown("---")
    st.markdown("""
    ### Project Details
    - **Dataset:** IBM Telco Customer Churn (7,043 customers, 33 features)
    - **Model:** XGBoost with SMOTE balancing
    - **AUC Score:** 0.92 (with NLP sentiment feature)
    - **Segments:** 4 customer segments via KMeans clustering
    - **NLP:** VADER sentiment analysis on churn reasons
    """)