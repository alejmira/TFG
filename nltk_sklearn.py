# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 12:47:11 2023

@author: aleja
"""

import json
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


# Función para cargar los datos en JSON a una lista de diccionarios de python
# Los parámetros a y b denotan el rango de ficheros a cargar, que todos tienen de nombre datos*
def cargar_datos(a, b):
    datos = []
    for i in range(a, b+1):
        for line in open("datos" + str(i), "r"):
            datos.append(json.loads(line))
    return datos


# Función anláloga a la anterior para cargar los datos en JSON sin contar tweets repetidos
def cargar_datos_sin_repes(a, b):
    datos = []
    tweets = set()
    for i in range(a, b+1):
        for line in open("datos" + str(i), "r"):
            try:
                tweet = json.loads(line)["retweeted_status"]["extended_tweet"]["full_text"]
                if tweet not in tweets:
                    datos.append(json.loads(line))
                tweets.add(tweet)
            except:
                tweet= json.loads(line)["text"]
                if tweet not in tweets:
                    datos.append(json.loads(line))
                tweets.add(tweet)
    return list(datos)    


# Función de limpieza de texto para filtrar los tweets y quedarnos con los tweets en español
def filtrarEspañol(datos):
    for tweet in datos:
        if tweet["lang"] != "es":
            datos.remove(tweet)


# Función para mostrar los textos de los tweets en una lista
def show_tweets(datos, filtrar_español = True):
    if filtrar_español:
        filtrarEspañol(datos)
    result = []
    for i in datos:
        try:
            result.append(i["retweeted_status"]["extended_tweet"]["full_text"])
        except:
            result.append(i["text"])
    return result    


# Función que realiza el preprocesado de texto sobre un tweet tomado como string
# Esto es, primero tokeniza el tweet, luego elimina las stop-words, y finalmente realiza el stemming
def preprocess_text(text):
    tokenizer = TweetTokenizer()
    es_stop = set(stopwords.words('spanish'))
    es_stop |= {"navidad", "rt", "si", "usted", "toda", "todas", "q", "hacer", "voy", "solo", "ser", "pa", "y", "a", "o"} # Quitamos palabras que aparecen demasiado
    p_stemmer = SnowballStemmer("spanish")
    tokens = tokenizer.tokenize(text)
    tokens_lower = [i.lower() for i in tokens]
    stopped_tokens = [i for i in tokens_lower if not i in es_stop and i.isalpha()] 
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return stemmed_tokens


# Función que realiza el preprocesado de texto sobre un cuerpo de tweets
def preprocess_data(datos):
    filtrarEspañol(datos)
    preprocessed_data = []
    for tweet in datos:
        try:
            preprocessed_data.append(preprocess_text(tweet["retweeted_status"]["extended_tweet"]["full_text"]))
        except:
            preprocessed_data.append(preprocess_text(tweet["text"]))
    return preprocessed_data


# Función que realiza la extracción de características de un cuerpo al que se le ha realizado un limpiado de datos
# Por defecto utiliza el modelo BoW, pero se puede cambiar para que utilice el tfidf
# Devuelve la matriz de extracción de características, y un diccionario fe que muestra la relación entre columnas y palabras
def feature_extractor(doc_clean, mode = "BoW"):
    tweets_join = map(lambda x: " ".join(x), doc_clean)
    if mode == "BoW":
        dictionary = CountVectorizer(max_df = 1.0, min_df = 1)
    elif mode == "tfidf":
        dictionary = TfidfVectorizer(max_df = 1.0, min_df = 1)
    return dictionary, dictionary.fit_transform(tweets_join)


# Función que realiza el modelado de temas usando el algoritmo LSA con un diccionario y matriz de extracción de características dados, 
# un número de topics fijo y un número de palabras para mostrar por topic
# Utiliza una función auxiliar print_sklearn_topics para mostrar los topics de forma más estética
# Devuelve un objeto modelo de sklearn
# Se fija el random state a 0 para que no cambie y se pueda comparar pero en realidad da igual
def lsa(dictionary, matrix, number_of_topics, number_of_words):
    LSA = TruncatedSVD(n_components = number_of_topics)
    LSA.fit(matrix)
    print_sklearn_topics(dictionary, LSA, number_of_topics, number_of_words)
    return LSA


# Función que realiza el modelado de temas usando el algoritmo LDA con un diccionario y matriz de extracción de características dados, 
# un número de topics fijo y un número de palabras para mostrar por topic
# Utiliza una función auxiliar print_sklearn_topics para mostrar los topics de forma más estética
# Devuelve un objeto modelo de sklearn
# Se fija el random state a 0 para que no cambie y se pueda comparar pero en realidad da igual
def lda(dictionary, matrix, number_of_topics, number_of_words):
    LDA = LatentDirichletAllocation(n_components = number_of_topics, random_state=0)
    LDA.fit(matrix)  
    print_sklearn_topics(dictionary, LDA, number_of_topics, number_of_words)
    return LDA


# Función que realiza el modelado de temas usando el algoritmo NNMF con un diccionario y matriz de extracción de características dados, 
# un número de topics fijo y un número de palabras para mostrar por topic
# Utiliza una función auxiliar print_sklearn_topics para mostrar los topics de forma más estética
# Devuelve un objeto modelo de sklearn
# Se fija el random state a 0 para que no cambie y se pueda comparar pero en realidad da igual
def nmf(dictionary, matrix, number_of_topics, number_of_words):
    NNMF = NMF(n_components = number_of_topics, random_state=0)
    NNMF.fit(matrix)   
    print_sklearn_topics(dictionary, NNMF, number_of_topics, number_of_words)
    return NNMF


# Función que dado un tweet del cuerpo, la matriz de la extracción de características, y un objeto modelo de gensim, 
# devuelve a qué topic pertenecen estos tweets
def predict(a, b, tweets, matrix, model):
    transform = model.fit_transform(matrix)
    for i in range(a, b+1):
        print(tweets[i])
        print(transform[i])
        print()
        

# Función auxiliar para mostrar los temas de forma más estética
def print_sklearn_topics(dictionary, model, num_topics, num_words):
    for i in range(num_topics):
        print("topic " + str(i+1) + ":")
        indices = reversed(model.components_[i].argsort()[-num_words:])
        print([(dictionary.get_feature_names()[j], round(model.components_[i][j], 2)) for j in indices])
    

# Función para calcular los valores de coherencia de los topics dado un algoritmo f,
# un diccionario y matriz de la extracción de características, y el cuerpo de los textos de los tweets,
# en un rango de start a stop con paso step, que tienen los valores 2, 10 y 1 por defecto respectivamente
# f = LsiModel, f = LdaMulticore, f = Nmf
def compute_coherence_values_sklearn(f, dictionary, dictionary_gensim, matrix, doc_clean, mode = "BoW", start=2, stop=10, step=1):
    coherence_values = []
    for number_of_topics in range(start, stop+1, step):
        model = f(n_components=number_of_topics, random_state=0)
        model.fit(matrix)
        topics = []
        for i in range(number_of_topics):
            #indices = reversed(model.components_[i].argsort()[-number_of_words:]) # meter parámetro number_of_words = 20 si se quiere usar esto
            indices = reversed(model.components_[i].argsort()[-8:])
            topics.append([dictionary.get_feature_names()[j] for j in indices])
        coherencemodel = CoherenceModel(topics = topics, texts=doc_clean, dictionary = dictionary_gensim, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return coherence_values


# Función para representar gráficamente los valores de coherencia de los topics para un algoritmo dado,
# el cuerpo de los textos de los tweets, un modo de extracción de características que está por defecto el BoW, 
# en un rango de start a stop con paso step, que tienen los valores 2, 10 y 1 por defecto respectivamente
def plot_coherence(algorithm, doc_clean, mode = "BoW", start = 2, stop = 10, step = 1):
    dictionary, matrix=feature_extractor(doc_clean, mode)
    dictionary_gensim = corpora.Dictionary(doc_clean)
    if algorithm == "LSA":
        f = TruncatedSVD
    elif algorithm == "LDA":
        f = LatentDirichletAllocation
    elif algorithm == "NMF":
        f = NMF
    coherence_values = compute_coherence_values_sklearn(f, dictionary, dictionary_gensim, matrix, doc_clean, mode, start, stop, step)
    # Show graph
    x = range(start, stop+1, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show() 
       

# Función que utiliza las funciones anteriores para realizar el modelado de temas dados unos datos, 
# un algoritmo, un modo de extracción de características, el número de topics,
# y cuántas palabras por topic usar    
def main(datos, algorithm, mode, number_of_topics, number_of_words):
    clean = preprocess_data(datos)
    dictionary, matrix = feature_extractor(clean, mode)
    if algorithm == "LSA":
        model = lsa(dictionary, matrix, number_of_topics, number_of_words)
    elif algorithm == "LDA":
        model = lda(dictionary, matrix, number_of_topics, number_of_words)
    elif algorithm == "NMF":
        model = nmf(dictionary, matrix, number_of_topics, number_of_words)
    return clean, dictionary, matrix, model        