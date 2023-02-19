# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:14:53 2022

@author: aleja
"""

import json
import spacy
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.models import LdaMulticore
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

nlp = spacy.load('es_core_news_lg') # 'es_core_news_sm' 'es_core_news_md' 'es_core_news_lg'


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
# Esto es, primero tokeniza el tweet, luego se lematizan, y finalmente se eliminan las stop-words
def preprocess_text(text):
    tokens = nlp(text)
    es_stop = nlp.Defaults.stop_words
    es_stop |= {"navidad", "rt", "si", "usted", "toda", "todas", "q", "hacer", "voy", "solo", "ser", "pa", "y", "a", "o"} # Quitamos palabras que aparecen demasiado
    lemmatized_tokens = [i.lemma_.lower() for i in tokens]
    tokens_without_emoji_and_punct = [i for i in lemmatized_tokens if i.isalpha() or i.isnumeric()]
    stopped_tokens = [i for i in tokens_without_emoji_and_punct if not i in es_stop]
    return stopped_tokens           
   
    
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
# Devuelve la matriz de extracción de características, y un diccionario que muestra la relación entre columnas y palabras
def feature_extractor(doc_clean, mode = "BoW"):
    dictionary = corpora.Dictionary(doc_clean)
    matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    if mode == "tfidf":
        matrix = TfidfModel(matrix)[matrix]
    return dictionary, matrix


# Función que realiza el modelado de temas usando el algoritmo LSA con un diccionario y matriz de extracción de características dados, 
# un número de topics fijo y un número de palabras para mostrar por topic
# Utiliza una función auxiliar print_gensim_topics para mostrar los topics de forma más estética
# Devuelve un objeto modelo de gensim
def lsa(dictionary, matrix, number_of_topics, number_of_words):
    lsamodel = LsiModel(matrix, num_topics=number_of_topics, id2word = dictionary)
    #print(lsamodel.print_topics(num_topics=number_of_topics, num_words=number_of_words))
    print_gensim_topics(lsamodel, number_of_topics, number_of_words)
    return lsamodel


# Función que realiza el modelado de temas usando el algoritmo LDA con un diccionario y matriz de extracción de características dados, 
# un número de topics fijo y un número de palabras para mostrar por topic
# Utiliza una función auxiliar print_gensim_topics para mostrar los topics de forma más estética
# Devuelve un objeto modelo de gensim
# Se fija el random state a 0 para que no cambie y se puedan replicar ejecuciones
def lda(dictionary, matrix, number_of_topics, number_of_words):
    ldamodel = LdaMulticore(matrix, num_topics=number_of_topics, id2word=dictionary, random_state=0)
    #print(ldamodel.print_topics(num_topics=number_of_topics, num_words=number_of_words))
    print_gensim_topics(ldamodel, number_of_topics, number_of_words)
    return ldamodel


# Función que realiza el modelado de temas usando el algoritmo NNMF con un diccionario y matriz de extracción de características dados, 
# un número de topics fijo y un número de palabras para mostrar por topic
# Utiliza una función auxiliar print_gensim_topics para mostrar los topics de forma más estética
# Devuelve un objeto modelo lsa de gensim
def nmf(dictionary, matrix, number_of_topics, number_of_words):
    nmfmodel = Nmf(matrix, num_topics=number_of_topics, id2word=dictionary, random_state = 0)
    #print(nmfmodel.print_topics(num_topics=number_of_topics, num_words=number_of_words))
    print_gensim_topics(nmfmodel, number_of_topics, number_of_words)
    return nmfmodel


# Función que dado un tweet del cuerpo, la matriz de la extracción de características, y un objeto modelo de gensim, 
# devuelve a qué topic pertenecen estos tweets
def predict(a, b, tweets, matrix, model):
    for i in range(a, b+1):
        print(tweets[i])
        print(model[matrix[i]])
        print()


# Función auxiliar para mostrar los temas de forma más estética
def print_gensim_topics(model, num_topics, num_words):
    for i in model.print_topics(num_topics,num_words):
        print("topic " + str(i[0] + 1) + ":")
        L = i[1].split(" + ")
        R = []
        for j in L: 
            S = j.split("*")
            R.append((S[1][1:-1], float(S[0])))
        print(R)


# Función para calcular los valores de coherencia de los topics dado un algoritmo f,
# un diccionario y matriz de la extracción de características, y el cuerpo de los textos de los tweets,
# en un rango de start a stop con paso step, que tienen los valores 2, 10 y 1 por defecto respectivamente
# f = LsiModel, f = LdaMulticore, f = Nmf
def compute_coherence_values_gensim(f, dictionary, matrix, doc_clean, mode = "BoW", start=2, stop=10, step=1):
    coherence_values = []
    for number_of_topics in range(start, stop+1, step):
        model = f(matrix, num_topics=number_of_topics, id2word = dictionary)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return coherence_values


# Función para representar gráficamente los valores de coherencia de los topics para un algoritmo dado,
# el cuerpo de los textos de los tweets, un modo de extracción de características que está por defecto el BoW, 
# en un rango de start a stop con paso step, que tienen los valores 2, 10 y 1 por defecto respectivamente
def plot_coherence(algorithm, doc_clean, mode = "BoW", start = 2, stop = 10, step = 1):
    dictionary, matrix = feature_extractor(doc_clean, mode)
    if algorithm == "LSA":
        f = LsiModel
    elif algorithm == "LDA":
        f = LdaMulticore
    elif algorithm == "NMF":
        f = Nmf
    coherence_values = compute_coherence_values_gensim(f, dictionary, matrix,doc_clean, mode, start, stop, step)
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



