# ================================================
# 1. Import Required Libraries
# ================================================

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

import warnings
warnings.filterwarnings('ignore')

# ================================================
# 2. Load Dataset
# ================================================
df = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/IBMHRANALYTICS/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# print("Shape of dataset:", df.shape)
df.head()

# ================================================
# 3. Data Cleaning
# ================================================
# Drop irrelevant columns
df = df.drop(['EmployeeCount','EmployeeNumber','StandardHours','Over18'], axis=1)

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# ================================================
# 4. Encode Categorical Variables
# ================================================
# cat_cols = df.select_dtypes(include='object').columns
# le = LabelEncoder()
# for col in cat_cols:
#     df[col] = le.fit_transform(df[col])

# Encode categorical variables
df_encoded = df.copy()
cat_cols = df_encoded.select_dtypes(include='object').columns

encoders = {}   # Store LabelEncoders here
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le
print("Categorical columns encoded.")

# ================================================
# 5. SQL Integration using MYSQL
# ================================================

# engine = create_engine("mysql+pymysql://root:Abhi@100982@localhost/hr_analytics")
engine = create_engine("mysql+pymysql://root:Abhi%40100982@localhost/hr_analytics")


df.to_sql("employee_attrition", engine, index=False, if_exists="replace")
print("Data loaded into MySQL")

# Example SQL Queries
queries = {
    "Overall Attrition Rate": """
        SELECT 
            SUM(CASE WHEN Attrition=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS AttritionRate
        FROM employee_attrition;
    """,

    "Attrition by Department": """
        SELECT Department, 
               SUM(CASE WHEN Attrition=1 THEN 1 ELSE 0 END) AS AttritionCount,
               COUNT(*) AS TotalEmployees,
               (SUM(CASE WHEN Attrition=1 THEN 1 ELSE 0 END)*100.0/COUNT(*)) AS AttritionRate
        FROM employee_attrition
        GROUP BY Department
        ORDER BY AttritionRate DESC;
    """,

    "Attrition by Gender": """
        SELECT Gender, 
               SUM(CASE WHEN Attrition=1 THEN 1 ELSE 0 END) AS AttritionCount,
               COUNT(*) AS TotalEmployees,
               (SUM(CASE WHEN Attrition=1 THEN 1 ELSE 0 END)*100.0/COUNT(*)) AS AttritionRate
        FROM employee_attrition
        GROUP BY Gender;
    """
}
    
for title, q in queries.items():
    print(f"\n--- {title} ---")
    result = pd.read_sql(q, engine)
    print(result)


# ================================================
# 6. Exploratory Data Analysis
# ================================================
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ================================================
# 7. Machine Learning Models
# ================================================
X = df_encoded.drop("Attrition", axis=1)
y = df_encoded["Attrition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ================================================
# 8. Feature Importance (Random Forest)
# ================================================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh', figsize=(8,6))
plt.title("Top 15 Important Features Driving Attrition")
plt.show()

# ================================================
# 9. Export to Excel for Business Users
# ================================================
with pd.ExcelWriter("HR_Attrition_Analysis.xlsx") as writer:
    df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    pd.read_sql(queries["Attrition by Department"], engine).to_excel(writer, sheet_name="Dept_Attrition", index=False)
    pd.read_sql(queries["Attrition by Gender"], engine).to_excel(writer, sheet_name="Gender_Attrition", index=False)
    pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"]).to_excel(writer, sheet_name="Model_Accuracy")

print("Excel file 'HR_Attrition_Analysis.xlsx' generated successfully!")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="HR Analytics - Employee Attrition", layout="wide")

# ================================================
# 1. Load Data
# ================================================
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/IBMHRANALYTICS/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = df.drop(['EmployeeCount','EmployeeNumber','StandardHours','Over18'], axis=1)
    return df

df = load_data()
st.title("HR Analytics & Attrition Prediction Dashboard")
st.write("Analyze employee attrition patterns and predict future attrition risk.")

# ================================================
# 2. Data Overview
# ================================================
st.subheader("Dataset Overview")
st.write(df.head())
st.write(f"Shape of dataset: {df.shape}")
st.write(f"Missing values: {df.isnull().sum().sum()}")

# ================================================
# 3. Encode categorical variables
# ================================================
# Encode categorical variables
df_encoded = df.copy()
cat_cols = df_encoded.select_dtypes(include='object').columns

encoders = {}   # Store LabelEncoders here
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

# ================================================
# 4. Exploratory Data Analysis
# ================================================
st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("### Attrition Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Attrition", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df_encoded.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# Attrition by Department
st.write("### Attrition by Department")
attrition_by_dept = df.groupby("Department")["Attrition"].value_counts(normalize=True).mul(100).rename("AttritionRate").reset_index()
st.dataframe(attrition_by_dept)

# ================================================
# 5. Machine Learning Models
# ================================================
st.subheader("Machine Learning Models - Predict Attrition")

X = df_encoded.drop("Attrition", axis=1)
y = df_encoded["Attrition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

st.write("### Model Performance")
st.dataframe(pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"]))

# ================================================
# 6. Feature Importance
# ================================================
st.subheader("Top Features Driving Attrition (Random Forest)")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feat_importances.nlargest(10)

fig, ax = plt.subplots()
top_features.plot(kind="barh", ax=ax)
st.pyplot(fig)

# ================================================
# 7. Prediction for New Employee
# ================================================
st.subheader("Predict Attrition for a New Employee")

input_data = {}
for col in df.drop("Attrition", axis=1).columns:
    if df[col].dtype == "object":
        input_data[col] = st.selectbox(f"{col}", df[col].unique())
    else:
        input_data[col] = st.number_input(
            f"{col}", 
            float(df[col].min()), 
            float(df[col].max()), 
            float(df[col].mean())
        )

if st.button("Predict"):
    new_df = pd.DataFrame([input_data])

    # Use stored encoders (don’t re-fit!)
    for col in [c for c in cat_cols if c != "Attrition"]:
        new_df[col] = encoders[col].transform(new_df[col])

    # Scale numerical values
    new_scaled = scaler.transform(new_df)

    # Predict with Random Forest
    prediction = rf.predict(new_scaled)[0]
    st.success(f"Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}")
