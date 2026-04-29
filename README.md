## IBM HR Analytics — Employee Attrition & Performance
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?style=flat&logo=mysql)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![Excel](https://img.shields.io/badge/Excel-Export-217346?style=flat&logo=microsoftexcel)

An end-to-end HR analytics pipeline that combines **SQL querying, Exploratory Data Analysis, Machine Learning, and an interactive Streamlit dashboard** to analyse and predict employee attrition using the IBM HR Analytics dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Model Accuracy Results](#model-accuracy-results)
- [Key Insights](#key-insights)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Project Overview

This project analyses the **IBM HR Analytics Employee Attrition & Performance Dataset** to uncover key factors influencing why employees leave an organisation.
The pipeline covers:
- Data cleaning and preprocessing
- SQL-based analytical queries via MySQL
- Exploratory Data Analysis (EDA) with visualisations
- Four ML classification models with accuracy benchmarking
- Feature importance analysis (top attrition drivers)
- An interactive Streamlit dashboard for HR teams with live prediction

## Dataset

| Property | Detail |
| :--- | :--- |
| **Source** | IBM HR Analytics (fictional dataset by IBM data scientists) |
| **Rows** | 1,470 employees |
| **Total Columns** | 35 features |
| **Target Variable** | `Attrition` (Yes / No) |
| **Attrition Rate** | ~16% (238 out of 1,470 employees) |

**Key Features Include:**
Age, Department, DistanceFromHome, Education, EnvironmentSatisfaction, Gender, JobInvolvement, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, NumCompaniesWorked, OverTime, PerformanceRating, WorkLifeBalance, YearsAtCompany

**Dropped Columns (irrelevant):**
`EmployeeCount`, `EmployeeNumber`, `StandardHours`, `Over18`

## Technologies Used

| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, cleaning, manipulation |
| **NumPy** | latest | Numerical operations |
| **Matplotlib** | latest | Static visualisations |
| **Seaborn** | latest | Statistical plots and heatmaps |
| **Scikit-Learn** | 1.x | ML models, encoding, scaling, metrics |
| **MySQL** | 8.0 | SQL-based attrition queries |
| **SQLAlchemy** | latest | Python–MySQL ORM connection |
| **Streamlit** | 1.x | Interactive web dashboard |
| **OpenPyXL / Excel** | latest | Business-friendly Excel export |

### Python Libraries (from source code)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
```

## Project Workflow
```
Raw CSV Dataset
      │
      ▼
Data Cleaning & Preprocessing
(Drop irrelevant cols → Label Encode → StandardScaler)
      │
      ▼
SQL Integration (MySQL via SQLAlchemy)
(Overall attrition rate, by Department, by Gender)
      │
      ▼
Exploratory Data Analysis (EDA)
(Attrition distribution, Correlation Heatmap, Dept breakdown)
      │
      ▼
Machine Learning Models
(Logistic Regression / Decision Tree / Random Forest / Gradient Boosting)
      │
      ▼
Feature Importance Analysis
(Top 15 drivers of attrition via Random Forest)
      │
      ▼
Excel Export + Streamlit Dashboard
(Business reports + Interactive HR prediction tool)
```

## Model Accuracy Results

All four models were trained on an **80/20 stratified train-test split** with `random_state=42`. Features were scaled using `StandardScaler`. Target: `Attrition` (binary classification).

| Model | Accuracy | Notes |
| :--- | :---: | :--- |
| **Logistic Regression** | ~79% | Good baseline, interpretable |
| **Decision Tree** | ~81% | `max_depth=5`, fast training |
| **Random Forest** | **~83%** | Best overall — `n_estimators=200` |
| **Gradient Boosting** | ~83% | Strong performance, slower training |

> **Random Forest** was selected as the primary prediction model for the Streamlit dashboard due to its consistent accuracy and interpretable feature importances.

### Classification Report (Random Forest — approximate)

| Class | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| No Attrition (0) | ~0.88 | ~0.94 | ~0.91 |
| Attrition (1) | ~0.68 | ~0.50 | ~0.57 |
| **Overall Accuracy** | | | **~83%** |

> Note: Class imbalance (~84% No, ~16% Yes) affects recall for the minority attrition class.

### SQL Analytical Results

| Query | Result |
| :--- | :--- |
| **Overall Attrition Rate** | ~16.1% |
| **Highest Attrition Dept** | Sales (~20.6%) |
| **Attrition by Gender** | Male: ~17%, Female: ~14.8% |

## Key Insights
- **OverTime** is the single strongest predictor of attrition — employees working overtime are significantly more likely to leave
- **MonthlyIncome** — lower-income employees show higher attrition rates
- **JobSatisfaction** and **WorkLifeBalance** — low scores strongly correlate with attrition
- **YearsAtCompany** — attrition peaks in the first 1–3 years (onboarding risk window)
- **Sales department** has the highest attrition rate (~20%) among all departments
- **Age** — younger employees (25–35) are more likely to leave than senior employees

### Top 10 Features Driving Attrition (Random Forest)

| Rank | Feature |
| :---: | :--- |
| 1 | OverTime |
| 2 | MonthlyIncome |
| 3 | Age |
| 4 | TotalWorkingYears |
| 5 | DailyRate |
| 6 | MonthlyRate |
| 7 | DistanceFromHome |
| 8 | YearsAtCompany |
| 9 | JobSatisfaction |
| 10 | WorkLifeBalance |

## Installation and Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/IBM-HR-Analytics-Employee-Attrition-Performance.git
cd IBM-HR-Analytics-Employee-Attrition-Performance
```
### Step 2 — Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit sqlalchemy pymysql openpyxl
```
### Step 3 — Configure MySQL
1. Start your MySQL server
2. Create a database named `hr_analytics`
3. Update the connection string in the code:
```python
engine = create_engine("mysql+pymysql://your_user:your_password@localhost/hr_analytics")
```
### Step 4 — Update Dataset Path
Replace the hardcoded path with your local path or place the CSV in the project folder:
```python
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
```
### Step 5 — Run the Analysis Script
```bash
python hr_attrition_analysis.py
```
### Step 6 — Launch the Streamlit Dashboard
```bash
streamlit run hr_attrition_streamlit.py
```
Open your browser at `http://localhost:8501`

## Usage
1. **Dashboard loads** with dataset overview (shape, missing values, sample rows)
2. **EDA section** displays attrition distribution and correlation heatmap side-by-side
3. **Attrition by Department** shows normalised rates interactively
4. **Model Accuracy table** compares all 4 ML models
5. **Feature Importance chart** shows top 10 attrition drivers
6. **Prediction form** — fill in employee details, click **Predict** to get Attrition / No Attrition result

### Excel Export
Running the analysis script generates `HR_Attrition_Analysis.xlsx` with 4 sheets:
- `Cleaned_Data` — preprocessed dataset
- `Dept_Attrition` — attrition rates by department
- `Gender_Attrition` — attrition rates by gender
- `Model_Accuracy` — all model accuracy scores

## Conclusion
This project provides actionable insights into employee attrition and demonstrates a complete **EDA → SQL → ML → Dashboard** pipeline. The final solution enables HR teams to:
- Identify high-risk attrition patterns by department, gender, and job role
- Understand the top drivers of attrition (OverTime, Income, Satisfaction)
- Predict attrition risk for individual employees in real time via the Streamlit dashboard
- Export findings to Excel for business reporting

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
