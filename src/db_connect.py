import psycopg2
from sqlalchemy import create_engine
import pandas as pd

#connection details

DB_CONFIG = {
    'host': 'localhost',
    'port': '5433',
    'database': 'churn_project',
    'user': 'postgres',
    'password': 'admin123'
}

#Test basic connection

conn = psycopg2.connect(**DB_CONFIG)
print("Connection successful!")
conn.close()

#Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

print("sqlalchemy engine created successfully!")

#Test by loading the csv and pushing to postgresql
df=pd.read_csv("data/raw/Telco_customer_churn.csv")
print(f"\n Dataset loaded-shape:{df.shape}")

df.to_sql("telco_raw", engine, if_exists="replace", index=False)
print("Data pushed to PostgreSQL successfully!: telco_raw table created")


#verfiy it loaded correctly
result = pd.read_sql("SELECT COUNT(*) as total_rows FROM telco_raw", engine)
print(f" Rows in database: {result['total_rows'][0]}")