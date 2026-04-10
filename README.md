# Customer Segmentation & Churn Prediction

An end-to-end data science project predicting telecom customer churn 
using machine learning, NLP, and business intelligence.

## Live Demo
🚀 [Streamlit App](your-streamlit-url-here)

## Project Overview
- **Dataset:** IBM Telco Customer Churn (7,043 customers, 33 features)
- **Model:** XGBoost — AUC: 0.92
- **Segments:** 4 customer segments via KMeans clustering
- **NLP:** VADER sentiment analysis on churn reasons (+5.58% AUC improvement)

## Key Findings
- 47% of new customers (0-12 months) churn — highest risk window
- Competitor issues (33%) and poor customer service (31%) drive 64% of churn
- Adding NLP sentiment feature improved model AUC from 0.8675 to 0.9233
- High Risk segment: 2,410 customers with $87,581/month revenue at risk

## Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| PostgreSQL | Database & SQL feature engineering |
| pandas / numpy | Data manipulation |
| scikit-learn | ML preprocessing & evaluation |
| XGBoost | Churn prediction model |
| SHAP | Model explainability |
| NLTK / VADER | Sentiment analysis |
| Power BI | Business intelligence dashboard |
| Streamlit | Web app deployment |

## Project Structure
customer-churn-project/
├── notebooks/          # Jupyter notebooks (EDA, Segmentation, ML, NLP)
├── src/                # Python scripts
├── data/               # Raw, processed, predictions
├── models/             # Trained model files
├── dashboard/          # Power BI files
├── app/                # Streamlit application
└── sql/                # SQL feature engineering queries

## Results
| Metric | Value |
|---|---|
| Model AUC | 0.92 |
| Churn Rate | 26.54% |
| High Risk Customers | 2,410 |
| Monthly Revenue at Risk | $87,581 |
| NLP Improvement | +5.58% AUC |

## How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/customer-churn-project

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app_v2.py
```

