## IBM-HR-Analytics-Employee-Attrition-Performance
This project focuses on analyzing employee attrition using IBM HR Analytics dataset. The goal is to uncover factors influencing attrition, perform exploratory data analysis (EDA), integrate SQL queries, and build machine learning models for prediction. A Streamlit dashboard was also developed for interactive exploration.

## Project Overview
This project analyzes the **IBM HR Analytics Employee Attrition & Performance Dataset** to uncover key factors influencing employee attrition.  
The analysis includes **data cleaning, SQL queries, exploratory data analysis (EDA), machine learning modeling, and an interactive Streamlit dashboard**.

## Dataset
- **Source:** IBM HR Analytics dataset (fictional dataset created by IBM data scientists).
- **Rows:** 1470 employees  
- **Columns:** 35 features (demographics, job satisfaction, income, work-life balance, etc.)  
- **Target Variable:** Attrition (Yes/No)

## Tools & Technologies
- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn)
- **SQL (MySQL)** – to query attrition by department, gender, etc.
- **Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Streamlit** – Interactive dashboard for visualization & prediction
- **Excel Export** – Business-friendly reports

## Project Workflow
### 1. Data Preprocessing
- Dropped irrelevant columns (`EmployeeCount`, `EmployeeNumber`, `StandardHours`, `Over18`)
- Checked for missing values (none found)
- Encoded categorical variables using Label Encoding
- Scaled numerical features with `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
- Distribution of Attrition (16% attrition rate)
- Correlation Heatmap
- Attrition by Department, Gender, Job Role, etc.

### 3. SQL Queries
- Overall Attrition Rate
- Attrition by Department
- Attrition by Gender

### 4. Machine Learning Models
- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest (Best performance)**  
- **Gradient Boosting**  

Random Forest provided the most reliable results.  
Feature importance showed **OverTime, MonthlyIncome, JobSatisfaction, and WorkLifeBalance** as key drivers.

### 5. Streamlit Dashboard
- Dataset Overview
- Attrition Distribution & Correlation Heatmap
- Attrition by Department (interactive)
- Model Accuracy Comparison
- Predict Attrition for a New Employee (user input form)

## Results & Insights
- **Attrition Rate:** ~16% of employees left the company.  
- **Key Drivers of Attrition:** Overtime, low Job Satisfaction, low Monthly Income, poor Work-Life Balance.  
- **Model Accuracy:** Random Forest & Gradient Boosting performed the best.  
- **Streamlit Dashboard:** Provided HR teams with interactive tools for analysis and prediction.

## Learnings & Challenges
- Learned how to integrate **SQL + Python + ML + Streamlit** in a single pipeline.
- Handling categorical encoding while ensuring model interpretability was challenging.
- Hyperparameter tuning improved Random Forest accuracy but required careful balancing.
- Building an interactive Streamlit dashboard made insights accessible to non-technical users.

## Conclusion
This project provides actionable insights into employee attrition and demonstrates the power of combining **EDA, SQL, ML models, and dashboards**.  
The final solution can help HR teams **predict attrition risk** and **design better retention strategies**.
