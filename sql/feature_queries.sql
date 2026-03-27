
--EXPLORING THE RAW TABLE 

--get a feel for the data
SELECT * FROM telco_raw LIMIT 10;

--check churn distribution
SELECT "Churn Label", COUNT(*) as total,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS PERCENTAGE
FROM telco_raw
GROUP BY "Churn Label";

--TENURE GROUPS

-- group customers by how long they've been with the company
SELECT "customerID","Tenure Months",
    CASE
        WHEN "Tenure Months" BETWEEN 0 AND 12 THEN 'New'    --0-12 months
        WHEN "Tenure Months" BETWEEN 13 AND 36 THEN 'Mid'   -- 1-3 years
        WHEN "Tenure Months" BETWEEN 37 AND 60 THEN 'Loyal'  -- 3-5 years
        ELSE 'Champion'     -- 5+ years
    END AS tenure_group
FROM telco_raw
ORDER BY "Tenure Months";


-- REVENUE BANDS

--Segment customers by monthly spend
SELECT "customerID", "Monthly Charges",
    CASE
        WHEN "Monthly Charges" < 35 THEN 'Low'
        WHEN "Monthly Charges" < 65 THEN 'Medium'
        WHEN "Monthly Charges" < 85 THEN 'High'
        ELSE 'Premium'
    END AS revenue_band
FROM telco_raw
ORDER BY "Monthly Charges";


-- SERVICE COUNT(how many services each customer uses)

-- Count number of add-on services per customer
SELECT "customerID",
    (CASE WHEN "Phone Service"    = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Multiple Lines"   = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Internet Service" != 'No' THEN 1 ELSE 0 END +
     CASE WHEN "Online Security"  = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Online Backup"    = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Device Protection"= 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Tech Support"     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Streaming TV"     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Streaming Movies" = 'Yes' THEN 1 ELSE 0 END) AS service_count
FROM telco_raw
ORDER BY service_count DESC;



--CHURN RATE BY TENURE GROUP

-- Which tenure group churns the most?
SELECT
    CASE
        WHEN "Tenure Months" BETWEEN 0  AND 12 THEN 'New'
        WHEN "Tenure Months" BETWEEN 13 AND 36 THEN 'Mid'
        WHEN "Tenure Months" BETWEEN 37 AND 60 THEN 'Loyal'
        ELSE 'Champion'
    END AS tenure_group,
    COUNT(*) as total_customers,
    SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM telco_raw
GROUP BY tenure_group
ORDER BY churn_rate DESC;


--CHURN RATE BY REVENUE BAND

-- Do high spenders churn more?
SELECT
    CASE
        WHEN "Monthly Charges" < 35 THEN 'Low'
        WHEN "Monthly Charges" < 65 THEN 'Medium'
        WHEN "Monthly Charges" < 85 THEN 'High'
        ELSE 'Premium'
    END AS revenue_band,
    COUNT(*) as total_customers,
    ROUND(AVG("Monthly Charges")::NUMERIC, 2) as avg_monthly,
    SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM telco_raw
GROUP BY revenue_band
ORDER BY churn_rate DESC;




--MASTER FEATURE TABLE(combining all the features into one table)

-- This is the big one — creates all features in one table
-- We'll save this as a view in PostgreSQL
CREATE OR REPLACE VIEW customer_features AS
SELECT
    "customer ID",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract",
    "Payment Method",
    "Internet Service",
    "Churn Label",

    -- Tenure group
    CASE
        WHEN "Tenure Months" BETWEEN 0  AND 12 THEN 'New'
        WHEN "Tenure Months" BETWEEN 13 AND 36 THEN 'Mid'
        WHEN "Tenure Months" BETWEEN 37 AND 60 THEN 'Loyal'
        ELSE 'Champion'
    END AS tenure_group,

    -- Revenue band
    CASE
        WHEN "Monthly Charges" < 35 THEN 'Low'
        WHEN "Monthly Charges" < 65 THEN 'Medium'
        WHEN "Monthly Charges" < 85 THEN 'High'
        ELSE 'Premium'
    END AS revenue_band,

    -- Service count
    (CASE WHEN "Phone Service"    = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Multiple Lines"   = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Internet Service" != 'No' THEN 1 ELSE 0 END +
     CASE WHEN "Online Security"  = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Online Backup"    = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Device Protection"= 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Tech Support"     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Streaming TV"     = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN "Streaming Movies" = 'Yes' THEN 1 ELSE 0 END) AS service_count,

    -- High value flag
    CASE
        WHEN "Monthly Charges" >= 85 AND "Tenure Months" > 24 THEN 'High Value'
        ELSE 'Standard'
    END AS customer_value,

    -- Risk flag
    CASE
        WHEN "Contract" = 'Month-to-month'
         AND "Monthly Charges" > 65
         AND "Tenure Months" < 12 THEN 'High Risk'
        WHEN "Contract" = 'Month-to-month' THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_segment

FROM telco_raw;