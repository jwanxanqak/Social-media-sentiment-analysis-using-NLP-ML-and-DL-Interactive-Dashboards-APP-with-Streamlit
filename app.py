import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.models import CoherenceModel




st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🦅 ")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🦅 ")

DATA_URL = ("Tweets.csv")

@st.cache_data 
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()


########################################
#MODELO DE IDENTIFICACIÓN DE TOPICOS
########################################

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocesamiento de texto
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result

# Crear diccionario y corpus
data['processed_text'] = data['text'].map(preprocess)
dictionary = corpora.Dictionary(data['processed_text'])
corpus = [dictionary.doc2bow(text) for text in data['processed_text']]

# Definir el rango de hiperparámetros
num_topics_range = st.sidebar.slider('Número de Tópicos', 5, 20, (5, 15))
passes_range = st.sidebar.slider('Número de Pasadas', 5, 30, (10, 20))
iterations_range = st.sidebar.slider('Número de Iteraciones', 50, 200, (50, 150))

best_coherence = 0
best_model = None

# Grid Search para optimización de hiperparámetros
for num_topics in range(num_topics_range[0], num_topics_range[1] + 1, 5):
    for passes in range(passes_range[0], passes_range[1] + 1, 5):
        for iterations in range(iterations_range[0], iterations_range[1] + 50, 50):
            model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, iterations=iterations)
            coherence_model = CoherenceModel(model=model, texts=data['processed_text'], dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            if coherence_score > best_coherence:
                best_coherence = coherence_score
                best_model = model

# Mostrar el mejor modelo
st.sidebar.subheader("Mejor Modelo LDA")
best_topics = best_model.print_topics(num_words=4)
for topic in best_topics:
    st.write(topic)

# Mostrar la coherencia del mejor modelo
st.write(f"Coherencia del Mejor Modelo: {best_coherence:.4f}")

# Visualización de los tópicos
st.subheader("Visualización de Tópicos")
for i, topic in enumerate(best_topics):
    st.write(f"Tópico {i+1}: {topic}")

# Mostrar los parámetros del mejor modelo
st.subheader("Parámetros del Mejor Modelo")
st.write(f"Número de Tópicos: {best_model.num_topics}")
st.write(f"Pasadas: {passes}")
st.write(f"Iteraciones: {iterations}")

