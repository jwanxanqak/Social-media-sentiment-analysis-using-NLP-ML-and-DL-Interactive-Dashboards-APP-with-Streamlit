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
#Show random tweet
########################################


st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])


########################################
#geospatial representation of tweets
########################################

st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", False, key='2'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)


#####################################################
#Breakdown airline by sentiment: comparative analysis
#####################################################

st.sidebar.subheader("Breakdown airline by sentiment: comparative analysis")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=6)

if len(choice) > 0:
    st.subheader("Breakdown airline by sentiment: comparative analysis")
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='airline', y='airline_sentiment',
                         histfunc='count', color='airline_sentiment',  
                         facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
                          height=600, width=800)
    st.plotly_chart(fig_0)

########################################
#Word Cloud
########################################

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", False, key='7'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    st.pyplot(fig)

   

###########################################
#SENTIMENT CLASSIFICATION MODEL: ML
###########################################


# Text vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['text']).toarray()
y = data['airline_sentiment']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar to adjust hyperparameters
st.sidebar.header("Hyperparameter tuning")
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
max_iter = st.sidebar.slider("Max Iter", 100, 1000, 100)

# Train model with tuned hyperparameters
model = LogisticRegression(C=C, max_iter=max_iter)
model.fit(X_train, y_train)

# Predictions and report
y_pred = model.predict(X_test)
st.subheader("Reporte de Clasificación")
st.text(classification_report(y_test, y_pred))

# Hyperparameter optimization
if st.sidebar.button("Optimize Hyperparameters"):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [100, 200, 300, 400, 500]
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    st.write("Best Hyperparameters:", best_params)

    # Train model with better hyperparameters
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    st.subheader("Classification Report with Better Hyperparameters")
    st.text(classification_report(y_test, y_pred_best))


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

