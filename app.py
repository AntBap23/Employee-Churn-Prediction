

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title('📊 Employee Churn Prediction and Insights')

# --- Load Data and Model ---
@st.cache_data
def load_data():
    return pd.read_csv("HR_Data_engineered.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_insights():
    try:
        with open("insights.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "No insights found."

df = load_data()
model = load_model()
insight_text = load_insights()

# --- Sidebar Navigation ---
section = st.sidebar.radio("Navigate", ["Data Exploration", "Predict Churn", "Insights"])

# --- Section: Data Exploration ---
if section == "Data Exploration":
    st.header("🔍 Data Exploration")

    st.subheader("📌 Preview of Data")
    st.dataframe(df.head())

    st.subheader("📈 Target (Churn) Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='left', data=df, ax=ax)
    ax.set_xticklabels(['Stayed', 'Left'])
    ax.set_title("Employee Churn Distribution")
    st.pyplot(fig)

    st.subheader("📦 Boxplot: Satisfaction vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x='left', y='satisfaction_level', data=df, ax=ax)
    ax.set_xticklabels(['Stayed', 'Left'])
    ax.set_title("Satisfaction Level by Churn")
    st.pyplot(fig)

    st.subheader("🏢 Churn Rate by Department")
    department_cols = [col for col in df.columns if col.startswith('Department_')]
    dept_churn = {}
    for col in department_cols:
        dept = col.split('_', 1)[1]
        churn_rate = df[df[col] == True]['left'].mean()
        dept_churn[dept] = churn_rate
    dept_churn_series = pd.Series(dept_churn).sort_values(ascending=False)
    st.bar_chart(dept_churn_series)

    st.subheader("💰 Churn Rate by Salary Level")
    salary_churn = {
        'low': df[df['salary_low']]['left'].mean(),
        'medium': df[df['salary_medium']]['left'].mean(),
        'high': df[(~df['salary_low']) & (~df['salary_medium'])]['left'].mean()
    }
    st.bar_chart(pd.Series(salary_churn))

# --- Section: Predict Churn ---
elif section == "Predict Churn":
    st.header("🤖 Predict Employee Churn")

    with st.form("churn_form"):
        satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.slider('Last Evaluation', 0.0, 1.0, 0.7)
        number_project = st.number_input('Number of Projects', 2, 7, 4)
        average_monthly_hours = st.number_input('Average Monthly Hours', 96, 310, 180)
        time_spend_company = st.number_input('Years at Company', 1, 10, 3)
        work_accident = st.selectbox('Work Accident', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        promotion_last_5years = st.selectbox('Promotion in Last 5 Years', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        department = st.selectbox('Department', ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
        salary = st.selectbox('Salary', ['low', 'medium', 'high'])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_monthly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': work_accident,
            'promotion_last_5years': promotion_last_5years,
            'satisfaction_level_sq': satisfaction_level**2
        }

        # Encode department
        for d in ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management']:
            col_name = f"Department_{d}"
            input_data[col_name] = 1 if department == d else 0

        # Encode salary
        input_data['salary_low'] = 1 if salary == 'low' else 0
        input_data['salary_medium'] = 1 if salary == 'medium' else 0

        # Interaction features
        input_data['avg_hours_x_salary_low'] = average_monthly_hours if salary == 'low' else 0
        input_data['avg_hours_x_salary_medium'] = average_monthly_hours if salary == 'medium' else 0
        input_data['time_spend_x_salary_low'] = time_spend_company if salary == 'low' else 0
        input_data['time_spend_x_salary_medium'] = time_spend_company if salary == 'medium' else 0
        input_data['num_project_x_salary_low'] = number_project if salary == 'low' else 0
        input_data['num_project_x_salary_medium'] = number_project if salary == 'medium' else 0

        input_df = pd.DataFrame([input_data])

        # Align with model input features
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {'🚨 Will Churn' if prediction == 1 else '✅ Will Not Churn'}")
        st.write(f"Confidence: {prob:.2%}")

# --- Section: Insights ---
elif section == "Insights":
    st.header("🧠 Key Insights and Conclusions")

    if insight_text:
        for line in insight_text.splitlines():
            if line.startswith("Data Analysis Key Findings"):
                st.markdown("### 📊 Data Analysis Key Findings")
            elif line.startswith("Insights or Next Steps"):
                st.markdown("---\n### 🧭 Insights or Next Steps")
            elif line.startswith("-") or line.startswith("•"):
                st.markdown(f"- {line.lstrip('-• ')}")
            elif ":" in line:
                title, value = line.split(":", 1)
                st.markdown(f"**{title.strip()}**: {value.strip()}")
            else:
                st.markdown(line)
    else:
        st.info("No insights found.")


