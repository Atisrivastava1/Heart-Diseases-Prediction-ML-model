import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart_model.pkl')
scaler = joblib.load('scaler.pkl')
expected = joblib.load('columns.pkl')

st.title("Heart Disease Prediction App by Ati") 
st.markdown("This app predicts the likelihood of heart disease based on user input features. Please fill in the required information below and click 'Predict' to see the results.") 
age = st.slider ("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina(TA)", "Atypical Angina(AA)", "Non-Anginal Pain(NAP)", "Asymptomatic(ASYM)"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting Electrocardiographic(ECG) Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == "Male" else 0],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [1 if fbs == "Yes" else 0],
        'restecg': [restecg],
        'oldpeak': [oldpeak],
        'slope': [slope]
    })

    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data, columns=['cp', 'restecg', 'slope'], drop_first=True)

    # Ensure all expected columns are present
    for col in expected:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the model's expected input
    input_data = input_data[expected]

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease. However, please consult a healthcare professional for a comprehensive assessment.")
