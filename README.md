# 📊 Customer Segmentation & Churn Prediction

An end-to-end data science project that predicts telecom customer churn,
segments customers using RFM analysis, and provides actionable retention
strategies — built with Python, PostgreSQL, XGBoost, SHAP, NLTK, Power BI
and Streamlit.

---

## 🚀 Live Demo
👉 [Streamlit App](https://your-streamlit-url.streamlit.app)

---

## 📌 Project Overview

A telecom company has 7,043 customers and loses 26.54% of them every year.
This project answers three business questions:

- **Who** is most likely to churn?
- **Why** are they leaving?
- **What** can we do to keep them?

---

## 🔑 Key Results

| Metric | Value |
|---|---|
| Model AUC Score | 0.92 |
| Overall Churn Rate | 26.54% |
| High Risk Customers | 2,410 |
| Monthly Revenue at Risk | $87,581 |
| NLP AUC Improvement | +5.58% |
| Customer Segments | 4 |

---

## 💡 Key Findings

- **47%** of new customers (0–12 months tenure) churn — the most critical retention window
- **Competitor issues (33%)** and **poor customer service (31%)** drive 64% of all churn
- High-paying customers churn MORE — Premium segment ($98/month avg) has 33.9% churn rate
- Adding NLP sentiment as a feature improved model AUC from **0.8675 → 0.9233**
- Month-to-month contract customers churn at **5x the rate** of two-year contract customers
- Electronic check payment users have the **highest churn rate (45%)** of any payment method

---

## 🗂️ Project Structure
```

customer-churn-project/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_segmentation.ipynb     # RFM + KMeans Clustering
│   ├── 03_modeling.ipynb         # XGBoost Churn Prediction
│   ├── 04_nlp.ipynb              # VADER Sentiment Analysis
│   └── 05_powerbi_export.ipynb   # Power BI Data Export
├── src/
│   ├── db_connect.py             # PostgreSQL connection
│   └── config.py                 # Configuration
├── sql/
│   └── feature_queries.sql       # SQL feature engineering
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned + engineered features
│   └── predictions/              # Model predictions
├── models/
│   ├── xgb_model.pkl             # Trained XGBoost model
│   └── feature_names.pkl         # Feature names for Streamlit
├── dashboard/
│   └── churn_dashboard.pbix      # Power BI dashboard
├── app/
│   └── streamlit_app_v2.py       # Streamlit web application
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies
└── README.md

```
---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| PostgreSQL 17 | Database & SQL feature engineering |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Data visualisation |
| scikit-learn | ML preprocessing & evaluation |
| XGBoost | Churn prediction model |
| SHAP | Model explainability |
| NLTK / VADER | NLP sentiment analysis |
| Power BI | Business intelligence dashboard |
| Streamlit | Web app deployment |
| SQLAlchemy | Python-PostgreSQL bridge |
| python-dotenv | Secure credential management |

---

## 📊 Project Phases

**Phase 1 — Data & SQL Layer**
Downloaded the IBM Telco Customer Churn dataset (7,043 rows, 33 features)
and loaded it into PostgreSQL. Wrote SQL queries to engineer features
including tenure groups, revenue bands, service counts and risk segments.
Conducted full EDA with ydata-profiling.

**Phase 2 — Customer Segmentation**
Built RFM (Recency, Frequency, Monetary) scores adapted for Telco data.
Used KMeans clustering validated by Elbow method and Silhouette scores
to identify 4 customer segments: Champions, Loyal Basics, New Passives
and High Risk. K=4 achieved the highest Silhouette Score of 0.4267.

**Phase 3 — Churn Prediction**
Trained Logistic Regression, Random Forest and XGBoost models after
handling class imbalance with SMOTE. XGBoost achieved AUC of 0.8675.
Used SHAP for global feature importance and individual waterfall
explanations. Top churn drivers: Monthly Charges, Contract Type, RFM Score.

**Phase 4 — NLP Analysis**
Applied VADER sentiment analysis to the Churn Reason column (1,869
churned customers). Categorised reasons into 7 business themes.
Added sentiment score as an XGBoost feature — improved AUC from
0.8675 to 0.9233 (+5.58%). Key finding: competitor churn is neutral
in sentiment while product/offer churn is most negative (-0.494).

**Phase 5 — Power BI Dashboard**
Built a 4-page interactive dashboard covering Overview KPIs,
Segment Analysis, Churn Drivers and NLP Insights. Key metrics:
total customers, churn rate, revenue at risk, average CLTV.

**Phase 6 — Streamlit Deployment**
Deployed a real-time churn prediction web app with a 10-field
simplified input form, SHAP waterfall chart, risk gauge, and
personalised retention strategy recommendations.

**Phase 7 — Retention Analysis**
Cohort analysis, Kaplan-Meier survival curves, CLV analysis
and win-back strategy recommendations per segment.

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-project.git
cd customer-churn-project
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and fill in your PostgreSQL credentials
```

**5. Run the Streamlit app**
```bash
streamlit run app/streamlit_app_v2.py
```

---

## 📈 Model Performance

| Model | AUC Score | Accuracy |
|---|---|---|
| Logistic Regression | 0.8704 | 79% |
| XGBoost | 0.8675 | 81% |
| XGBoost + NLP | **0.9233** | **83%** |
| Random Forest | 0.8496 | 78% |

---

## 👥 Customer Segments

| Segment | Customers | Churn Rate | Strategy |
|---|---|---|---|
| High Risk | 2,410 (34%) | 47% | Immediate retention action |
| New Passives | 1,509 (21%) | 24% | Onboarding program |
| Champions | 2,152 (31%) | 16% | Loyalty rewards |
| Loyal Basics | 972 (14%) | 4% | Maintain quality |

---

## 📋 Churn Themes (NLP)

| Theme | Customers | % | Strategy |
|---|---|---|---|
| Competitor | 621 | 33.2% | Price-match + exclusive offers |
| Customer Service | 587 | 31.4% | Staff training + SLA improvement |
| Other | 255 | 13.6% | Exit interviews |
| Network Quality | 147 | 7.9% | Infrastructure upgrades |
| Product & Offers | 102 | 5.5% | Personalised upgrade offers |
| Pricing | 98 | 5.2% | Flexible plan options |
| Personal Reasons | 59 | 3.2% | Natural attrition |

---

## 📄 Dataset

**IBM Telco Customer Churn Dataset**
- Source: IBM Watson Analytics
- Rows: 7,043 customers
- Features: 33 columns
- Target: Churn Label (Yes/No)
- Special columns: CLTV, Churn Score, Churn Reason

---

## 🔐 Security Note

Database credentials are stored in a `.env` file and never committed
to GitHub. Copy `.env.example` to `.env` and fill in your own values.

---

