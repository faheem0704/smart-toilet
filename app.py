import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and pipelines
scaler = joblib.load("scaler.joblib")
pipeline_clf = joblib.load("classification_pipeline.joblib")
pipeline_reg = joblib.load("regression_pipeline.joblib")

clf_model = joblib.load("Random Forest_classification_model.joblib")
protein_model = joblib.load("Random Forest_protein_regression_model.joblib")
gravity_model = joblib.load("Random Forest_gravity_regression_model.joblib")

# Load dataset for default values
data = pd.read_csv(r"C:\Users\Elite\Desktop\uni\5th sem\ML\smartoiletproject\data\urinalysis_tests.csv")

st.title("Smart Toilet System: Health Prediction")

# User Inputs
st.header("Enter the following details:")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", options=["Male", "Female"])
glucose = st.number_input("Glucose", min_value=50.0, max_value=300.0, value=100.0)
ph = st.number_input("pH", min_value=4.0, max_value=8.0, value=6.5)
specific_gravity = st.number_input("Specific Gravity", min_value=1.0, max_value=1.03, value=1.02)
color = st.selectbox("Color", options=data['Color'].unique())

# Mapping and preparing input
gender_map = {'Male': 0, 'Female': 1}
user_input = {
    'Age': age,
    'Gender': gender_map[gender],
    'Glucose': glucose,
    'pH': ph,
    'Specific Gravity': specific_gravity,
    'Color': color
}

# Fill other features with dataset median/most frequent
default_data = data.median(numeric_only=True).to_dict()
default_data.update(data.mode().iloc[0].to_dict())  # Include mode for categorical
user_data = {**default_data, **user_input}

# Convert to DataFrame for model
user_df = pd.DataFrame([user_data])

# Apply scaler on features it was trained on
scaled_features = ['Glucose', 'pH', 'Specific Gravity']
user_df[scaled_features] = scaler.transform(user_df[scaled_features])

# Preprocess for classification
user_df_encoded = pipeline_clf.transform(user_df)

# Predictions
if st.button("Predict Diagnosis"):
    diagnosis_pred = clf_model.predict(user_df_encoded)
    st.subheader("Diagnosis Prediction:")
    st.write("**POSITIVE**" if diagnosis_pred[0] == 1 else "**NEGATIVE**")

if st.button("Predict Protein Level"):
    # Use data without Protein and Specific Gravity for this model
    user_df_reg = user_df.drop(['Protein', 'Specific Gravity'], axis=1)
    user_df_reg_encoded = pipeline_reg.transform(user_df_reg)
    protein_pred = protein_model.predict(user_df_reg_encoded)
    st.subheader("Protein Level Prediction:")
    st.write(f"Predicted Protein Level: {protein_pred[0]:.2f} g/dL")

if st.button("Predict Specific Gravity"):
    # Use data without Protein for Specific Gravity prediction
    user_df_gravity = user_df.drop(['Protein'], axis=1)
    user_df_gravity_encoded = pipeline_reg.transform(user_df_gravity)
    gravity_pred = gravity_model.predict(user_df_gravity_encoded)
    st.subheader("Specific Gravity Prediction:")
    st.write(f"Predicted Specific Gravity: {gravity_pred[0]:.4f}")

st.write("\n")
st.info("Models and pipelines loaded successfully.")
