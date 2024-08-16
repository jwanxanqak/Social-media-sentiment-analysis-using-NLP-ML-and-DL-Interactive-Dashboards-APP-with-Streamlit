import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from PIL import Image


# Load models
logistic_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('rf_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Function to load classification reports from .txt files
def load_report(file_path):
    with open(file_path, 'r') as file:
        report = file.read()
    return report

# Load classification reports
logistic_report = load_report('logistic_classification_report.txt')
rf_report = load_report('rf_classification_report.txt')
svm_report = load_report('svm_classification_report.txt')

# Function to load and display images
def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width=True)

# App title
st.title('Predicting sentiments of tweets about airlines')
st.write('Thank you God for everything...for your blessings')

# Individual predictions
st.header('Individual Prediction')
st.write('Instructions for use: The variables record predefined values ​​for making predictions using the prediction button. The values ​​can be modified to your preference.')

# Default values
default_values = {
    'tweet_id': '5.70002509906838e+17',
    'airline_sentiment_confidence': 1.0,
    'negativereason': 'nan',
    'negativereason_confidence': 'nan',
    'airline': 'Delta',
    'name': 'redbaronfilms',
    'retweet_count': 0,
    'text': '@JetBlue Our non-profit ARC would love tickets as we rely on airlines for extractions in saving abducted children and returning them Home!',
    'tweet_created': '2015-02-23 15:29:23-0800',
    'latitude': 28.7290894049044,
    'longitude': -105.784045961871
}

tweet_id = st.text_input('Tweet ID', value=default_values['tweet_id'])
airline_sentiment_confidence = st.number_input('Confianza del Sentimiento de la Aerolínea', value=default_values['airline_sentiment_confidence'])
negativereason = st.text_input('Razón Negativa', value=default_values['negativereason'])
negativereason_confidence = st.text_input('Confianza de la Razón Negativa', value=default_values['negativereason_confidence'])
airline = st.text_input('Aerolínea', value=default_values['airline'])
name = st.text_input('Nombre', value=default_values['name'])
retweet_count = st.number_input('Conteo de Retweets', min_value=0, value=default_values['retweet_count'])
text = st.text_area('Texto del Tweet', value=default_values['text'])
tweet_created = st.text_input('Fecha de Creación del Tweet', value=default_values['tweet_created'])
latitude = st.number_input('Latitud', value=default_values['latitude'], format="%.14f")
longitude = st.number_input('Longitud', value=default_values['longitude'], format="%.14f")


# Model selection
model_choice = st.selectbox('Choose the model', ('Logistic Regression', 'Random Forest', 'SVM'), key='1')

if st.button('Predict'):
    data = {
        'tweet_id': [tweet_id],
        'airline_sentiment_confidence': [airline_sentiment_confidence],
        'negativereason': [negativereason],
        'negativereason_confidence': [negativereason_confidence],
        'airline': [airline],
        'name': [name],
        'retweet_count': [retweet_count],
        'text': [text],
        'tweet_created': [tweet_created],
        'latitude': [latitude],
        'longitude': [longitude]
    }
    df = pd.DataFrame(data)
    
    if model_choice == 'Logistic Regression':
        model = logistic_model
        report = logistic_report
        confusion_matrix_image = 'logistic_regression_confusion_matrix.png'
        roc_curve_image = 'logistic_regression_roc_curve.png'
    elif model_choice == 'Random Forest':
        model = rf_model
        report = rf_report
        confusion_matrix_image = 'random_forest_confusion_matrix.png'
        roc_curve_image = 'random_forest_roc_curve.png'
    else:
        model = svm_model
        report = svm_report
        confusion_matrix_image = 'svm_confusion_matrix.png'
        roc_curve_image = 'svm_roc_curve.png'

    # Make prediction
    prediction = model.predict(df)
    st.write(f'Prediction: {prediction[0]}')

    # Show confusion matrix, ROC curve and Classification Report
    st.write('Confusion Matrix:')
    display_image(confusion_matrix_image)

    st.write('ROC Curve:')
    display_image(roc_curve_image)

    st.write('Classification Report:')
    st.text(report)

    # Save predictions to a CSV file
    df['prediction'] = prediction
    df.to_csv('predictions.csv', index=False)
    st.write('Predictions saved.csv')

    
# Upload file for batch predictions
st.header('Batch Prediction')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model_choice = st.selectbox('Choose the model', ('Logistic Regression', 'Random Forest', 'SVM'), key='2')

    if model_choice == 'Logistic Regression':
        model = logistic_model
        report_mass = logistic_report
        confusion_matrix_image = 'logistic_regression_confusion_matrix.png'
        roc_curve_image = 'logistic_regression_roc_curve.png'
    elif model_choice == 'Random Forest':
        model = rf_model
        report_mass = rf_report
        confusion_matrix_image = 'random_forest_confusion_matrix.png'
        roc_curve_image = 'random_forest_roc_curve.png'
    else:
        model = svm_model
        report_mass = svm_report
        confusion_matrix_image = 'svm_confusion_matrix.png'
        roc_curve_image = 'svm_roc_curve.png'

    predictions = model.predict(df)
    df['predictions'] = predictions
    st.write(df)
    df.to_csv('predictions.csv', index=False)
    st.download_button(label="Download Predictions", data=df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

    # Show confusion matrix, ROC curve and Classification Report
    st.write('Confusion Matrix:')
    display_image(confusion_matrix_image)

    st.write('ROC Curve:')
    display_image(roc_curve_image)

    st.write('Batch Classification Report:')
    st.text(report_mass)
