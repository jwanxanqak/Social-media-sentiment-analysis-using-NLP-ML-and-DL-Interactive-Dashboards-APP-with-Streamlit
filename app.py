import streamlit as st
import joblib
import pandas as pd

# Cargar los modelos
logistic_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('rf_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Título de la aplicación
st.title('Predicción de Sentimiento de Aerolíneas')

# Subir archivo para predicciones masivas
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model_choice = st.selectbox('Elige el modelo', ('Logistic Regression', 'Random Forest', 'SVM'))

    if model_choice == 'Logistic Regression':
        model = logistic_model
    elif model_choice == 'Random Forest':
        model = rf_model
    else:
        model = svm_model

    predictions = model.predict(df)
    df['predictions'] = predictions
    st.write(df)
    df.to_csv('predictions.csv', index=False)
    st.download_button(label="Descargar Predicciones", data=df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')
    
    
