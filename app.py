
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HeartCare | AI Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load('KNN_heart_model.pkl')
scaler = joblib.load('scaler.pkl')
expected = joblib.load('columns.pkl')

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #071426, #102b46, #0b1d32);
    color: white;
}

/* Remove default top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header */
.header {
    background: linear-gradient(120deg, #123b5c, #176b87);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.header h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.header p {
    color: #c9e5f5;
    font-size: 16px;
}

/* Section Cards */
.section-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
}

/* Section Headings */
.section-title {
    color: #54d6e8;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Input Labels */
label, .stSelectbox label, .stNumberInput label,
.stSlider label {
    color: #e7f3ff !important;
    font-weight: 600 !important;
}

/* Input Background */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background-color: #18344e !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #31536d !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding-top: 10px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #00b4d8, #0077b6);
    color: white;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0096c7, #023e8a);
    transform: translateY(-2px);
    color: white;
    border: none;
}

/* Result Cards */
.result-success {
    background: rgba(25, 180, 120, 0.15);
    border: 1px solid #21c58e;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    color: #6fffc2;
    margin-top: 25px;
}

.result-danger {
    background: rgba(255, 65, 85, 0.15);
    border: 1px solid #ff5268;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    color: #ff8795;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    color: #91aabd;
    font-size: 13px;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>❤️ HeartCare AI</h1>
    <p>Intelligent Heart Disease Prediction System</p>
    <p>Powered by Machine Learning | K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)


# ---------------- INTRO ----------------
st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
    <h3 style="color:#ffffff;">Welcome to Your Heart Health Dashboard</h3>
    <p style="color:#b9cddd;">
    Enter the patient's health parameters below to generate an
    AI-assisted prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT FORM ----------------
with st.form("heart_prediction_form"):

    st.markdown('<div class="section-title">👤 Personal Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (Years)", 18, 100, 25)

    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("---")

    st.markdown('<div class="section-title">🫀 Medical Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        cp = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina(TA)",
                "Atypical Angina(AA)",
                "Non-Anginal Pain(NAP)",
                "Asymptomatic(ASYM)"
            ]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            80, 200, 120
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            100, 600, 200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["Yes", "No"]
        )

    with col2:
        restecg = st.selectbox(
            "Resting ECG Results",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ]
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            0.0, 10.0, 1.0, step=0.1
        )

        slope = st.selectbox(
            "Peak Exercise ST Slope",
            ["Upsloping", "Flat", "Downsloping"]
        )

    st.markdown("---")

    submit = st.form_submit_button(
        "🔍  Predict Heart Disease",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submit:

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

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=['cp', 'restecg', 'slope'],
        drop_first=True
    )

    # Match training columns
    for col in expected:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected]

    # Scaling
    scaled_input = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_input)

    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:center;color:white;'>Prediction Results</h2>",
        unsafe_allow_html=True
    )

    if prediction[0] == 1:

        st.markdown("""
        <div class="result-danger">
            <h2>⚠️ Potential Heart Disease Risk</h2>
            <h4>The model predicts a positive result.</h4>
            <p>Please consult a qualified healthcare professional
            for proper medical evaluation.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-success">
            <h2>💚 Lower Predicted Risk</h2>
            <h4>The model predicts a negative result.</h4>
            <p>This does not rule out heart disease.
            Please consult a healthcare professional if you have concerns.</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Disclaimer: This application is an educational machine learning "
        "project and is not a medical diagnostic tool."
    )


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ by Ati, Shruti, Ritu & Deva</p>
    <p>HeartCare AI | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
