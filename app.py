# app.py

import streamlit as st
import pandas as pd
import joblib # Or pickle, depending on how the model was saved
import numpy as np # Import numpy for numerical operations

# Set the title of the application
st.title('Employee Churn Prediction')

# --- Model Loading ---
# Assuming the model and the list of selected features are saved.
try:
    # Replace with your actual model file name and features list file name
    model = joblib.load('churn_prediction_model.pkl')
    selected_features = joblib.load('selected_features.pkl')
except FileNotFoundError:
    st.error("Model file or selected features file not found. Please ensure 'churn_prediction_model.pkl' and 'selected_features.pkl' are in the same directory.")
    st.stop() # Stop the app if loading fails
except Exception as e:
    st.error(f"An error occurred while loading the model or features: {e}")
    st.stop()

# --- User Input ---
st.header("Employee Information")

# Create input fields for each original feature
# Map the feature names to user-friendly labels and appropriate widgets.
satisfaction_level = st.slider('Satisfaction Level', 0.0, 1.0, 0.5)
last_evaluation = st.slider('Last Evaluation', 0.0, 1.0, 0.7)
number_project = st.number_input('Number of Projects', 2, 7, 4)
average_montly_hours = st.number_input('Average Monthly Hours', 96, 310, 180) # Adjust range based on your data
time_spend_company = st.number_input('Time Spend in Company (years)', 2, 10, 3)
Work_accident = st.selectbox('Work Accident', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
promotion_last_5years = st.selectbox('Promotion in Last 5 Years', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Categorical features
department = st.selectbox('Department', ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.selectbox('Salary', ['low', 'medium', 'high'])

# --- Prediction Button ---
if st.button('Predict Churn'):
    # --- Data Preparation for Prediction ---
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[
        satisfaction_level,
        last_evaluation,
        number_project,
        average_montly_hours,
        time_spend_company,
        Work_accident,
        promotion_last_5years,
        department,
        salary
    ]], columns=['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company', 'Work_accident',
                'promotion_last_5years', 'Department', 'salary'])

    # Apply the exact same feature engineering steps as in the notebook

    # Create dummy variables for 'Department' and 'salary'
    # Need to handle potential missing dummy columns if a category is not selected by the user
    input_data = pd.get_dummies(input_data, columns=['Department', 'salary'], drop_first=True)

    # List of all expected dummy columns based on the training data
    # This list should be consistent with how dummy variables were created during training
    expected_department_dummies = [col for col in selected_features if col.startswith('Department_')]
    expected_salary_dummies = [col for col in selected_features if col.startswith('salary_')]

    # Add any missing dummy columns and fill with False (or 0 depending on get_dummies output type)
    for col in expected_department_dummies + expected_salary_dummies:
        if col not in input_data.columns:
            # Use the correct dtype based on get_dummies output (usually boolean or uint8)
            # Check the dtype of a dummy column in your training data if unsure
            input_data[col] = False # Assuming get_dummies produces boolean

    # Ensure the order of columns matches the training data before creating engineered features
    # This is important if interaction terms depend on the order or specific dummy column names
    # Let's first create the engineered features using the potentially reordered/completed input_data

    # Create polynomial feature for satisfaction_level
    input_data['satisfaction_level_sq'] = input_data['satisfaction_level']**2

    # Create interaction terms (ensure these match the ones used in training)
    # These columns must exist in input_data before creating interactions
    # The dummy columns were added above, so these interactions should now work
    input_data['avg_hours_x_salary_low'] = input_data['average_montly_hours'] * input_data['salary_low']
    input_data['avg_hours_x_salary_medium'] = input_data['average_montly_hours'] * input_data['salary_medium']

    input_data['time_spend_x_salary_low'] = input_data['time_spend_company'] * input_data['salary_low']
    input_data['time_spend_x_salary_medium'] = input_data['time_spend_company'] * input_data['salary_medium']

    input_data['num_project_x_salary_low'] = input_data['number_project'] * input_data['salary_low']
    input_data['num_project_x_salary_medium'] = input_data['number_project'] * input_data['salary_medium']


    # Finally, reorder input_data columns to precisely match the order of features the model was trained on
    # This is crucial for consistent predictions
    try:
        input_data = input_data[selected_features]
    except KeyError as e:
         st.error(f"Feature mismatch: Input data is missing a feature expected by the model: {e}. Expected features: {selected_features}. Input features: {input_data.columns.tolist()}")
         st.stop()


    # --- Make Prediction ---
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1] # Probability of churn

    # --- Display Results ---
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"This employee is likely to churn.")
    else:
        st.success(f"This employee is unlikely to churn.")

    st.write(f"Probability of churning: {prediction_proba[0]:.2f}")

    # Optional: Add an explanation of the prediction or factors influencing it
    # This might require more advanced techniques like SHAP or LIME