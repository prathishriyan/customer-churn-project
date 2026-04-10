import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model + pre-calculated data ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_rfm_data():
    """
    Load pre-calculated RFM scores and segments from customer_segments.csv.
    Confirmed columns: CustomerID, Recency, Frequency, Monetary,
    R_Score, F_Score, M_Score, RFM_Score, Cluster, Segment,
    Contract, tenure_group, risk_segment, PCA1, PCA2
    """
    df = pd.read_csv("data/processed/customer_segments.csv")
    return df


@st.cache_data
def load_rfm_boundaries():
    """
    Calculate quintile boundaries FROM the actual training dataset.
    Uses exact column names confirmed from customer_segments.csv:
      - "Tenure Months", "service_count", "Monthly Charges"
    """
    df = pd.read_csv("data/processed/customer_segments.csv")

    # ── Find each column by checking all variants ─────────────────────────
    def find_col(df, *candidates):
        """Return first candidate that exists as a column, else None."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # ── Your CSV has: Recency, Frequency, Monetary directly ─────────────
    # Recency  = 72 - Tenure Months (already computed in segmentation notebook)
    # Frequency = service_count
    # Monetary  = Monthly Charges
    recency_col  = find_col(df, "Recency",   "recency")
    freq_col     = find_col(df, "Frequency", "frequency")
    monetary_col = find_col(df, "Monetary",  "monetary")

    missing = []
    if recency_col  is None: missing.append("Recency")
    if freq_col     is None: missing.append("Frequency")
    if monetary_col is None: missing.append("Monetary")

    if missing:
        raise ValueError(
            f"Cannot find required columns: {missing}\n"
            f"Available columns in CSV: {df.columns.tolist()}"
        )

    boundaries = {
        # Recency is already 72-tenure so higher = newer customer
        "recency_q"   : df[recency_col].quantile([0.2, 0.4, 0.6, 0.8]).values,
        "frequency_q" : df[freq_col].quantile([0.2, 0.4, 0.6, 0.8]).values,
        "monetary_q"  : df[monetary_col].quantile([0.2, 0.4, 0.6, 0.8]).values,
        "recency_col" : recency_col,
        "freq_col"    : freq_col,
        "monetary_col": monetary_col,
    }
    return boundaries

model, feature_names = load_model()
rfm_df               = load_rfm_data()

# Show column debug info in sidebar during testing
try:
    boundaries = load_rfm_boundaries()
except ValueError as e:
    st.error(f"❌ RFM boundary error: {e}")
    st.info("Check that customer_segments.csv has: 'Tenure Months', 'service_count', 'Monthly Charges'")
    st.stop()

# ── RFM scoring using dataset quintile boundaries ─────────────────────────────
def compute_rfm_score(tenure, num_services, monthly_charges):
    """
    Score new customers using quintile boundaries from training data.
    Recency = 72 - tenure (matches how segmentation notebook computed it).
    Frequency = number of services.
    Monetary = monthly charges.
    """
    recency = 72 - tenure  # match segmentation notebook formula

    # R Score: higher recency = newer customer = higher score
    r_score = 1 + sum(recency > q for q in boundaries["recency_q"])

    # F Score: more services = higher score
    f_score = 1 + sum(num_services > q for q in boundaries["frequency_q"])

    # M Score: higher spend = higher score
    m_score = 1 + sum(monthly_charges > q for q in boundaries["monetary_q"])

    # Clamp to 1-5
    r_score = max(1, min(5, r_score))
    f_score = max(1, min(5, f_score))
    m_score = max(1, min(5, m_score))

    return int(r_score), int(f_score), int(m_score), int(r_score + f_score + m_score)


# ── Encode input and build feature dataframe ──────────────────────────────────
def build_input(tenure, monthly_charges, contract, internet, tech_support,
                online_security, num_services, senior, partner, dependents,
                payment, customer_rfm=None):
    """
    Build feature dataframe matching training feature set.
    If customer_rfm is provided (known customer), use pre-calculated scores.
    Otherwise compute from dataset quintile boundaries.
    """

    # ── Label encode maps (must match training LabelEncoder order) ──────────
    # These match the order sklearn LabelEncoder sees when trained on full data
    gender_map        = {"Female": 0, "Male": 1}
    yes_no_map        = {"No": 0, "Yes": 1}
    internet_map      = {"DSL": 0, "Fiber optic": 1, "No": 2}
    contract_map      = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map       = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)"  : 1,
        "Electronic check"         : 2,
        "Mailed check"             : 3,
    }
    tenure_group_map  = {"Champion": 0, "Loyal": 1, "Mid": 2, "New": 3}
    revenue_band_map  = {"High": 0, "Low": 1, "Medium": 2, "Premium": 3}
    segment_map       = {"Champions": 0, "High Risk": 1, "Loyal Basics": 2, "New Passives": 3}

    total_charges = monthly_charges * tenure

    # ── Tenure group & Revenue band ─────────────────────────────────────────
    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 36:
        tenure_group = "Mid"
    elif tenure <= 60:
        tenure_group = "Loyal"
    else:
        tenure_group = "Champion"

    if monthly_charges < 35:
        revenue_band = "Low"
    elif monthly_charges < 65:
        revenue_band = "Medium"
    elif monthly_charges < 85:
        revenue_band = "High"
    else:
        revenue_band = "Premium"

    # ── RFM Scores ──────────────────────────────────────────────────────────
    if customer_rfm is not None:
        # ✅ Known customer — use pre-calculated scores directly
        # Auto-detect column names from whatever was saved in CSV
        def _get(row, *keys):
            for k in keys:
                if k in row.index: return row[k]
            return None

        r_score   = int(_get(customer_rfm, "R_Score",   "r_score",   "RScore")   or 3)
        f_score   = int(_get(customer_rfm, "F_Score",   "f_score",   "FScore")   or 3)
        m_score   = int(_get(customer_rfm, "M_Score",   "m_score",   "MScore")   or 3)
        rfm_score = int(_get(customer_rfm, "RFM_Score", "rfm_score", "RFMScore") or 9)
        segment   = str(_get(customer_rfm, "Segment",   "segment")               or "Unknown")
    else:
        # ⚠️ New customer — calculate using dataset quintile boundaries
        r_score, f_score, m_score, rfm_score = compute_rfm_score(
            tenure, num_services, monthly_charges
        )
        # Estimate segment from rfm_score
        if rfm_score >= 13:
            segment = "Champions"
        elif rfm_score >= 10:
            segment = "Loyal Basics"
        elif rfm_score >= 7:
            segment = "New Passives"
        else:
            segment = "High Risk"

    # ── Customer value & risk segment ───────────────────────────────────────
    customer_value = 1 if (monthly_charges >= 85 and tenure > 24) else 0
    if contract == "Month-to-month" and monthly_charges > 65 and tenure < 12:
        risk_segment = 2
    elif contract == "Month-to-month":
        risk_segment = 1
    else:
        risk_segment = 0

    # ── Build raw dataframe ─────────────────────────────────────────────────
    data = {
        "Gender"            : gender_map["Male"],          # default Male (not collected in v2)
        "Senior Citizen"    : yes_no_map[senior],
        "Partner"           : yes_no_map[partner],
        "Dependents"        : yes_no_map[dependents],
        "Tenure Months"     : tenure,
        "Phone Service"     : yes_no_map["Yes"],
        "Multiple Lines"    : yes_no_map["No"],
        "Internet Service"  : internet_map[internet],
        "Online Security"   : yes_no_map[online_security],
        "Online Backup"     : yes_no_map["No"],
        "Device Protection" : yes_no_map["No"],
        "Tech Support"      : yes_no_map[tech_support],
        "Streaming TV"      : yes_no_map["No"],
        "Streaming Movies"  : yes_no_map["No"],
        "Contract"          : contract_map[contract],
        "Paperless Billing" : yes_no_map["Yes"],
        "Payment Method"    : payment_map[payment],
        "Monthly Charges"   : monthly_charges,
        "Total Charges"     : total_charges,
        "tenure_group"      : tenure_group_map[tenure_group],
        "revenue_band"      : revenue_band_map[revenue_band],
        "service_count"     : num_services,
        "customer_value"    : customer_value,
        "risk_segment"      : risk_segment,
        "Recency"           : 72 - tenure,
        "Frequency"         : num_services,
        "Monetary"          : monthly_charges,
        "R_Score"           : r_score,
        "F_Score"           : f_score,
        "M_Score"           : m_score,
        "RFM_Score"         : rfm_score,
        "Segment"           : segment_map.get(segment, 3),
        "Sentiment_Score"   : 0.0,
    }

    df = pd.DataFrame([data])

    # Add any missing columns as 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names], segment, rfm_score, r_score, f_score, m_score


# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict churn probability and get retention recommendations.")
st.divider()

# ── Optional Customer ID lookup ───────────────────────────────────────────────
with st.expander("🔍 Know the Customer ID? Look them up directly (more accurate)", expanded=False):
    customer_id_input = st.text_input(
        "Enter Customer ID (e.g. 7590-VHVEG)",
        help="If found, pre-calculated RFM scores from training data will be used — much more accurate!"
    )
    if customer_id_input:
        # Auto-detect CustomerID column name
        id_col = next((c for c in rfm_df.columns
                       if c.lower().replace(" ", "").replace("_", "") == "customerid"), None)
        seg_col     = next((c for c in rfm_df.columns if c.lower() == "segment"), "Segment")
        rfm_col     = next((c for c in rfm_df.columns if c.lower().replace("_","") == "rfmscore"), "RFM_Score")
        tenure_col2 = next((c for c in rfm_df.columns if c.lower().replace(" ","") == "tenuremonths"), "Tenure Months")
        charges_col2= next((c for c in rfm_df.columns if c.lower().replace(" ","") == "monthlycharges"), "Monthly Charges")

        if id_col is None:
            st.error(f"CustomerID column not found. Available: {rfm_df.columns.tolist()}")
            found_customer = None
        else:
            match = rfm_df[rfm_df[id_col].astype(str).str.upper() == customer_id_input.strip().upper()]
            if not match.empty:
                row = match.iloc[0]
                seg_col = next((c for c in rfm_df.columns if c.lower() == "segment"), "Segment")
                rfm_col = next((c for c in rfm_df.columns if c.lower() == "rfm_score"), "RFM_Score")
                st.success(f"✅ Customer found! Segment: **{row[seg_col]}** | RFM Score: **{row[rfm_col]}**")
                # Show available useful columns
                display_cols = [c for c in ["Recency","Frequency","Monetary","R_Score","F_Score","M_Score","RFM_Score","Segment","Contract"] if c in rfm_df.columns]
                st.dataframe(match[display_cols].reset_index(drop=True))
                found_customer = row
            else:
                st.warning("Customer ID not found — enter details manually below.")
                found_customer = None
    else:
        found_customer = None

# ── Manual input form ─────────────────────────────────────────────────────────
st.subheader("📋 Customer Details")
st.markdown("Fill in the customer information:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Info**")
    tenure          = st.number_input("How many months has customer been with us?",
                                       min_value=0, max_value=72, value=12,
                                       help="Number of months since they joined")
    contract        = st.selectbox("Contract Type",
                                    ["Month-to-month", "One year", "Two year"],
                                    help="Month-to-month = highest churn risk")
    monthly_charges = st.number_input("Monthly Bill ($)",
                                       min_value=18.0, max_value=120.0, value=65.0,
                                       help="How much they pay per month")

with col2:
    st.markdown("**Services**")
    internet        = st.selectbox("Internet Service",
                                    ["Fiber optic", "DSL", "No"],
                                    help="Fiber optic customers churn most")
    tech_support    = st.selectbox("Has Tech Support?", ["Yes", "No"],
                                    help="Customers with tech support churn less")
    online_security = st.selectbox("Has Online Security?", ["Yes", "No"],
                                    help="Security add-on reduces churn")
    num_services    = st.slider("Total number of services subscribed", 1, 9, 3,
                                 help="Phone, internet, streaming, backup etc.")

with col3:
    st.markdown("**Demographics**")
    senior     = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner    = st.selectbox("Has Partner?",    ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    payment    = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"],
                               help="Electronic check = highest churn risk")

predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
if predict_btn:

    input_df, segment, rfm_score, r_score, f_score, m_score = build_input(
        tenure, monthly_charges, contract, internet, tech_support,
        online_security, num_services, senior, partner, dependents,
        payment, customer_rfm=found_customer
    )

    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = 1 if churn_prob >= 0.5 else 0

    # ── Risk level ────────────────────────────────────────────────────────
    if churn_prob >= 0.7:
        risk_color = "#FF6B6B"
        risk_label = "HIGH RISK"
        risk_msg   = "Immediate retention action needed!"
        risk_icon  = "🔴"
    elif churn_prob >= 0.4:
        risk_color = "#ED7D31"
        risk_label = "MEDIUM RISK"
        risk_msg   = "Monitor closely and send retention offer."
        risk_icon  = "🟠"
    else:
        risk_color = "#70AD47"
        risk_label = "LOW RISK"
        risk_msg   = "Customer is stable. Maintain service quality."
        risk_icon  = "🟢"

    st.divider()

    # ── Result banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='padding:24px; border-radius:12px;
                background:{risk_color}22; border:2px solid {risk_color};
                text-align:center'>
        <h1 style='color:{risk_color}; margin:0'>{churn_prob*100:.1f}%</h1>
        <h2 style='color:{risk_color}; margin:8px 0'>{risk_icon} {risk_label}</h2>
        <p style='color:gray; margin:0'>{risk_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn Probability",  f"{churn_prob*100:.1f}%")
    m2.metric("Prediction",         "Will Churn ⚠️" if churn_pred else "Will Stay ✅")
    m3.metric("Customer Segment",   segment)
    m4.metric("Contract Type",      contract)

    # ── RFM debug info (helpful during testing) ───────────────────────────
    with st.expander("🧮 RFM Score Details (for testing/debugging)", expanded=False):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("R Score (Recency)",   r_score,   help="1-5, lower = newer customer")
        r2.metric("F Score (Frequency)", f_score,   help="1-5, higher = more services")
        r3.metric("M Score (Monetary)",  m_score,   help="1-5, higher = higher spend")
        r4.metric("RFM Total",           rfm_score, help="3-15, higher = better customer")
        st.caption(
            "✅ Using pre-calculated RFM from training data" if found_customer is not None
            else "⚠️ New customer — RFM estimated using dataset quintile boundaries"
        )

    st.divider()

    # ── SHAP + Strategy ───────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader("🔍 Why is this prediction made?")
        st.caption("Each bar shows how much that feature pushed the prediction toward or away from churn")
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.waterfall_plot(
                shap.Explanation(
                    values        = shap_values[0],
                    base_values   = explainer.expected_value,
                    data          = input_df.iloc[0],
                    feature_names = feature_names
                ), show=False
            )
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP chart unavailable: {e}")

    with right:
        st.subheader("💡 What should we do?")

        if churn_prob >= 0.7:
            st.error("🚨 Immediate Action Required")
            st.markdown("""
            - 📞 Call customer within **48 hours**
            - 💰 Offer **15–20% discount** for 3 months
            - 👤 Assign dedicated account manager
            """)
        elif churn_prob >= 0.4:
            st.warning("⚠️ Retention Offer Recommended")
            st.markdown("""
            - 🎁 Send **loyalty reward** or upgrade offer
            - 📧 Enrol in **satisfaction survey**
            - 📋 Add to monthly **watch list**
            """)
        else:
            st.success("✅ Customer is Stable")
            st.markdown("""
            - ⭐ Consider for **referral program**
            - 🔄 Maintain current service quality
            - 📊 Review in **next quarter**
            """)

        st.markdown("---")
        st.markdown("**Contract-specific advice:**")
        if contract == "Month-to-month":
            st.info("📋 Offer annual contract with **10% discount** to improve retention")
        elif contract == "One year":
            st.info("⬆️ Upgrade to **two-year contract** with loyalty bonus")
        else:
            st.success("✅ Already on two-year contract — lowest churn risk")

        if tenure < 12:
            st.warning("🤝 New customer — enrol in **onboarding program** (check-ins at month 3, 6, 9)")
        if monthly_charges > 85:
            st.info("💎 Premium customer — ensure **VIP support** experience")

    st.divider()

    # ── Risk factor summary ───────────────────────────────────────────────
    st.subheader("⚠️ Risk Factor Summary")

    factors = {
        "Contract Type"   : ("🔴 High Risk" if contract == "Month-to-month" else "🟢 Low Risk",
                             "Month-to-month contracts churn 5× more than 2-year"),
        "Tenure"          : ("🔴 High Risk" if tenure < 12 else
                             "🟡 Medium"   if tenure < 36 else "🟢 Low Risk",
                             f"{tenure} months — {'New customer, critical window' if tenure < 12 else 'Established customer'}"),
        "Monthly Charges" : ("🔴 High Risk" if monthly_charges > 65 else "🟢 Low Risk",
                             f"${monthly_charges:.2f}/month"),
        "Internet Service": ("🔴 High Risk" if internet == "Fiber optic" else "🟢 Low Risk",
                             f"{internet}"),
        "Tech Support"    : ("🟢 Low Risk" if tech_support == "Yes" else "🔴 High Risk",
                             "Tech support reduces churn significantly"),
        "Payment Method"  : ("🔴 High Risk" if payment == "Electronic check" else "🟢 Low Risk",
                             f"{payment}"),
    }

    cols = st.columns(3)
    for i, (factor, (level, explanation)) in enumerate(factors.items()):
        with cols[i % 3]:
            st.markdown(f"**{factor}**")
            st.markdown(level)
            st.caption(explanation)
            st.markdown("")

# ── Welcome screen ────────────────────────────────────────────────────────────
else:
    st.markdown("""
    ## How this tool works 👋
    This tool helps **retention teams** identify customers likely to cancel
    their subscription — before it happens.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
        **Step 1 — Enter Details**
        Fill in the customer's account information from your CRM system.
        Only 10 key fields needed.
        """)
    with c2:
        st.warning("""
        **Step 2 — Get Prediction**
        The AI model calculates churn probability based on 7,043 historical customers.
        """)
    with c3:
        st.success("""
        **Step 3 — Take Action**
        Follow the personalised retention strategy recommendations shown.
        """)

    st.markdown("---")
    st.subheader("📊 Model Performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Model AUC Score",     "0.92")
    p2.metric("Training Customers",  "7,043")
    p3.metric("Features Used",       "21")
    p4.metric("Segments Identified", "4")

    st.markdown("---")
    st.subheader("🎯 Key Churn Insights from Data")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.error("**47% churn rate**\namong new customers\n(0–12 months tenure)")
    with i2:
        st.warning("**35% churn rate**\namong high spenders\n($65–85/month)")
    with i3:
        st.success("**Only 4% churn rate**\namong Loyal Basics\n(long tenure, low spend)")