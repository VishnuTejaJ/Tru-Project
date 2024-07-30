import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from PIL import Image
import math

# from sklearn.metrics import DistanceMetric

# Define a stub if you know what it should look like
# Define euclidean function

# def EuclideanDistance(point1, point2):
#     distance = 0
#     for i in range(len(point1)):
#         distance += (point1[i] - point2[i]) ** 2
#     return math.sqrt(distance)

# Add the missing class to the current module's namespace
# DistanceMetric.EuclideanDistance = EuclideanDistance



# Define header function

def header():
    st.markdown("""
    <style>
    .header {
        font-size:30px;
        font-weight:bold;
        color:#4CAF50;
        text-align:center;
        padding: 10px;
    }
    </style>
    <div class="header">
        Cardio-Vascular Disease Prediction
    </div>
    """, unsafe_allow_html=True)

# Define footer function

def footer():
    st.markdown("""
    <style>
    .footer {
        font-size:18px;
        font-weight:bold;
        color:#4CAF50;
        text-align:center;
        padding: 15px;
        position:fixed;
        bottom:0;
        width:100%;
        background-color:#f1f1f1;
    }
    </style>
    <div class="footer">
        © 2024 Cardio-Vascular Disease Prediction. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


def show_page(page_name):
    st.session_state.page = page_name


def home_page():
    st.write("""
        ## Introduction

        Welcome to the Cardio-Vascular Disease Prediction! Our application is designed to help you assess your risk of cardiovascular diseases based on various health factors.
        
        ### What We Offer
        Our app provides a comprehensive analysis of several critical health indicators including:
        - **Age**: How your age influences your risk.  
        - **Gender**: The impact of gender on cardiovascular health.  
        - **Cholesterol Levels**: Understanding the role of cholesterol in heart disease.  
        - **Glucose Levels**: How glucose levels affect your cardiovascular risk.  
        - **Smoking Habits**: The effects of smoking on heart health.  
        - **Alcohol Consumption**: The role of alcohol in cardiovascular disease.  
        - **Physical Activity**: How your activity level influences your heart health.  
        - **BMI (Body Mass Index)**: The relationship between BMI and cardiovascular risk.  
        - **MAP (Mean Arterial Pressure)**: Understanding the significance of MAP in heart health.

        ### How to Use
        To use the app, simply enter your health details, and our advanced predictive models will analyze your data to provide an assessment. The app leverages cutting-edge machine learning techniques to offer accurate predictions and valuable insights.

        ### Getting Started
        Navigate through the app using the sidebar to explore different features, make predictions, and learn more about cardiovascular health. Whether you’re here to get a prediction or to understand how various factors influence your heart health, our app aims to provide you with the information you need to make informed decisions.
        ##### Thank you for choosing our app. We are committed to helping you understand and manage your cardiovascular health effectively!

    """)


        
# Model Prediction

def predict_the_answer(age, bmi, maps, gender, cholestrol, glucose, smoke, alco, active):
    gender = 1 if gender == 'M' else 2
    smoke = 1 if smoke == 'Yes' else 0
    cholestrol = 1 if cholestrol == 'Low' else 2 if cholestrol == 'Medium' else 3
    glucose = 1 if glucose == 'Low' else 2 if glucose == 'Medium' else 3  
    alco = 1 if alco == 'Yes' else 2
    active = 1 if active == 'Yes' else 2
    input_features = [1,age, bmi, maps, gender, cholestrol, glucose, smoke, alco, active]
    models = [xg_boost_1,knn_model, rf_model, svm_model, dt_model, ada_model]
    predictions = [model.predict([input_features])[0] for model in models]
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction


# Define prediction page content

def prediction_page():
    st.write('Please fill in the following details to get a prediction.')
    
    age = st.number_input('Enter your Age:', min_value=0.0, max_value=100.0, step=1.0)
    if st.checkbox('How age influences Cardio'):
        st.image('images/age vs variables.png')
    bmi = st.number_input('Enter BMI:', min_value=15.0, max_value=47.00, step=0.1)
    if st.checkbox('How BMI influences Cardio'):
        st.image('images/bmi vs var.png')
    maps = st.number_input('Enter MAP:', min_value=70.0, max_value=125.0, step=1.0)
    if st.checkbox('How MAP influences Cardio'):
        st.image('images/map vs var.png')
    gender = st.radio('Gender:', ['M', 'F'])
    if st.checkbox('How Gender influences Cardio'):
        st.image('images/gender vs var.png')
    cholestrol = st.radio('Cholesterol:', ['Low', 'Medium', 'High'])
    if st.checkbox('How Cholesterol influences Cardio'):
        st.image('images/cholesterol vs var.png')
    glucose = st.radio('Glucose:', ['Low', 'Medium', 'High'])
    if st.checkbox('How Glucose influences Cardio'):
        st.image('images/glucose vs var.png')
    smoke = st.radio('Smoke:', ['Yes', 'No'])
    if st.checkbox('How Smoke influences Cardio'):
        st.image('images/smoke vs var.png')
    alco = st.radio('Alcohol Consumption:', ['Yes', 'No'])
    if st.checkbox('How Alcohol influences Cardio'):
        st.image('images/alco vs var.png')
    active = st.radio('Physical Activity Level:', ['Yes', 'No'])
    if st.checkbox('How Physical Activity influences Cardio'):
        st.image('images/active vs var.png')

    if st.button('Predict'):
        if age and bmi and maps and cholestrol and glucose and alco and active:
            result = predict_the_answer(age, bmi, maps, gender, cholestrol, glucose, smoke, alco, active)
            if result == 1:
                st.error('You need Cardio')
            else:
                st.success('You are healthy')
            
    if st.checkbox('## Know more about this'):
        st.write('# Know more about Cardio')
        st.image('images/cardio.png')
        
        if st.checkbox('### Want to know how each factor is influencing Cardio?'):
            if st.checkbox('Cardio V/S Gender'):
                st.image('images/cardio_wrt_gender.png')
            if st.checkbox('Cardio V/S Cholesterol'):
                st.image('images/cardio_wrt_cholesterol.png')
            if st.checkbox('Glucose vs Cardio'):
                st.image('images/glucose_vs_cardio.png')
            if st.checkbox('Smoke vs Cardio'):
                st.image('images/smoke_vs_cardio.png')
            if st.checkbox('Alcohol vs Cardio'):
                st.image('images/alcohol_vs_cardio.png')
            if st.checkbox('Active vs Cardio'):
                st.image('images/active_vs_cardio.png')
    
        if st.checkbox('### How every Feature is related to Cardio'):
            st.image('images/features_heatmap.png')
    
        if st.checkbox('How age is related to cardio w.r.t gender'):
            st.image('images/cardio_age_gender.png')
    
        if st.checkbox('How age is influencing BMI w.r.t cardio'):
            st.image('images/age_vs_bmi_wrt_cardio.png')
    
        if st.checkbox('How smoke is influencing cardio w.r.t age'):
            st.image('images/smoke_vs_age_wrt_cardio.png')
    
        if st.checkbox('How Alcohol is influencing cardio w.r.t age'):
            st.image('images/alco_vs_age_cardio.png')
    
        if st.checkbox('How Gender BMI influences on gender w.r.t cardio'):
            st.image('images/gen_vs_bmi_cardio.png')

    if st.button("Back to Home"):
        show_page('Home')

# Define About Us page content

def about_us_page():
    html_content = """
    <div style="text-align: center;">
        <h1>Welcome to Nexus Tech Solutions!</h1>
        <p>At Nexus Tech Solutions, we are passionate about driving the digital revolution. Our mission is to empower businesses with cutting-edge machine learning and deep learning solutions. We specialize in transforming data into actionable insights to help you make informed decisions and achieve your goals.</p>
        <p>Our team of experts is dedicated to delivering innovative solutions tailored to your needs. Whether you're looking to optimize your operations, enhance user engagement, or explore new opportunities in the digital world, we're here to help.</p>
        <p>Join us on this exciting journey and let's shape the future together!</p>
    </div>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)

    if st.button("Back to Home"):
        show_page('Home')

# Load models
joblib_in_1 = open('models/xg_model_1', 'rb')
joblib_in_2 = open('models/knn_model', 'rb')
joblib_in_3 = open('models/rf_model', 'rb')
joblib_in_4 = open('models/svm_model', 'rb')
joblib_in_5 = open('models/dt_model', 'rb')
joblib_in_6 = open('models/ada_model', 'rb')

xg_boost_1 = joblib.load(joblib_in_1)
knn_model = joblib.load(joblib_in_2)
rf_model = joblib.load(joblib_in_3)
svm_model = joblib.load(joblib_in_4)
dt_model = joblib.load(joblib_in_5)
ada_model = joblib.load(joblib_in_6)


import streamlit as st

# Initialize session state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'  # Set a default value for 'page'

# Now you can safely access st.session_state.page
if st.session_state.page == 'Home':
    # Your code for the Home page
    st.write("Welcome to the Home page")


# Initialize session state for navigation if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Render header
header()

col1, col2, col3 = st.columns([1,1,1])


with col1:
        if st.button('Home') :
            show_page('Home')

with col2:
        if st.button('Prediction'):
            show_page('Prediction')
with col3 :
        if st.button('About Us'):
            show_page('About Us')


# Display the selected page
if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Prediction':
    prediction_page()
elif st.session_state.page == 'About Us':
    about_us_page()

# Render footer
footer()