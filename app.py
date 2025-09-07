import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

# Title and description
st.title("Heart Disease Prediction System")
st.markdown("""
This application predicts the likelihood of heart disease based on various medical parameters.
Please fill in all the required information below.
""")

# Load model and preprocessor function
@st.cache_resource
def load_model_and_preprocessor():
    try:
        # Load the saved model
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load the original dataset
        df = pd.read_csv('heart.csv')
        
        # Recreate the exact same preprocessor as in training
        categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal'] 
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        # Create pipelines
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore')) 
        ])
        
        # Create the exact same preprocessor as in training
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features),
                ('passthrough', 'passthrough', ['sex', 'fbs', 'exang']) 
            ],
            remainder='drop' 
        )
        
        # Fit the preprocessor on the original training data (without target)
        X = df.drop('target', axis=1)
        preprocessor.fit(X)
        
        return model, preprocessor
    except FileNotFoundError as e:
        if 'heart.csv' in str(e):
            st.error("Dataset file 'heart.csv' not found. Please ensure the dataset is in the same directory.")
        else:
            st.error("Model file 'best_model.pkl' not found. Please ensure the model is trained and saved.")
        return None, None

# Load model and preprocessor
model, preprocessor = load_model_and_preprocessor()

if model is None or preprocessor is None:
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Basic Information")
    
    # Age
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50,
        step=1,
        help="Patient's age in years"
    )
    
    # Sex
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="0: Female, 1: Male"
    )
    
    # Chest Pain Type (cp) - Categorical
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina", 
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )
    
    # Resting Blood Pressure
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120,
        step=1,
        help="Resting blood pressure in mm Hg"
    )
    
    # Serum Cholesterol
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200,
        step=1,
        help="Serum cholesterol level in mg/dl"
    )
    
    # Fasting Blood Sugar
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1 if fasting blood sugar > 120 mg/dl, 0 otherwise"
    )

with col2:
    st.header("🏥 Medical Tests")
    
    # Resting ECG (restecg) - Categorical
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality", 
            2: "Left Ventricular Hypertrophy"
        }[x],
        help="Resting electrocardiographic results"
    )
    
    # Maximum Heart Rate
    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150,
        step=1,
        help="Maximum heart rate achieved during exercise"
    )
    
    # Exercise Induced Angina
    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="1 if exercise induced angina, 0 otherwise"
    )
    
    # Oldpeak
    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        format="%.1f",
        help="ST depression induced by exercise relative to rest"
    )
    
    # Slope (slope) - Categorical
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x],
        help="The slope of the peak exercise ST segment"
    )
    
    # Number of Major Vessels (ca) - Categorical
    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        options=[0, 1, 2, 3, 4],
        help="Number of major vessels (0-4) colored by fluoroscopy"
    )
    
    # Thalassemia (thal) - Categorical
    thal = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversible Defect"
        }[x],
        help="Thalassemia type"
    )

# Prediction section
st.header("Prediction")

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    # Create input dataframe with exact same column order as training
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    try:
        # Transform the input data using the fitted preprocessor
        X_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == '1' or prediction == 1:
                st.error("⚠️ **HIGH RISK**")
                st.error("The model predicts that the patient **HAS** heart disease")
                risk_prob = probability[1] if len(probability) > 1 else probability[0]
                st.error(f"**Risk Probability: {risk_prob:.1%}**")
            else:
                st.success("✅ **LOW RISK**")
                st.success("The model predicts that the patient **DOES NOT HAVE** heart disease")
                safe_prob = probability[0] if len(probability) > 1 else (1 - probability[0])
                st.success(f"**Safe Probability: {safe_prob:.1%}**")
        
        with col2:
            # Display probability breakdown
            st.subheader("Probability Breakdown")
            if len(probability) > 1:
                st.write(f"No Heart Disease: {probability[0]:.1%}")
                st.write(f"Heart Disease: {probability[1]:.1%}")
            else:
                st.write(f"Probability: {probability[0]:.1%}")
        
        # Add disclaimer
        st.warning("""
        **Disclaimer:** This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice. 
        Please consult with a healthcare professional for proper medical evaluation and diagnosis.
        """)
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.write("**Patient Information:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"• Age: {age} years")
                st.write(f"• Sex: {'Male' if sex == 1 else 'Female'}")
                st.write(f"• Chest Pain: {['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][cp]}")
                st.write(f"• Resting BP: {trestbps} mm Hg")
                st.write(f"• Cholesterol: {chol} mg/dl")
                st.write(f"• Fasting Blood Sugar > 120: {'Yes' if fbs == 1 else 'No'}")
            
            with col2:
                st.write(f"• Resting ECG: {['Normal', 'ST-T Wave Abnormality', 'LV Hypertrophy'][restecg]}")
                st.write(f"• Max Heart Rate: {thalach}")
                st.write(f"• Exercise Angina: {'Yes' if exang == 1 else 'No'}")
                st.write(f"• ST Depression: {oldpeak}")
                st.write(f"• Slope: {['Upsloping', 'Flat', 'Downsloping'][slope]}")
                st.write(f"• Major Vessels: {ca}")
                st.write(f"• Thalassemia: {['Unknown', 'Normal', 'Fixed Defect', 'Reversible Defect'][thal]}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check that all inputs are valid and try again.")