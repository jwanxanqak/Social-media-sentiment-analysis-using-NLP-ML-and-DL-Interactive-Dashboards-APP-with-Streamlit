import streamlit as st
import joblib
import pandas as pd

# Cargar los modelos
logistic_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('rf_model.pkl')
svm_model = joblib.load('svm_model.pkl')
