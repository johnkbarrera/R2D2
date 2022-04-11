#!/usr/bin/env python
# coding: utf-8

# In[2]:




# !pip install -U gensim
# !pip install -U spacy
# !pip install nltk
# !pip install pyLDAvis
# !conda install pyplot

# !python -m spacy download es_core_news_sm
# !python -m spacy link es_core_news_sm es


# In[3]:


#!conda install -c plotly plotly=5.6.0


# In[4]:


import os
import pandas as pd
import numpy as np
import math
from datetime import datetime
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import plotly.express as px
from difflib import SequenceMatcher


# In[5]:


# Reclamos
file_name = 'data/data_merged.csv'
data_to_process = pd.read_pickle(file_name)


# In[6]:


# Filtramos solo nivel 1
data_to_process = data_to_process[data_to_process['Nivel']=='1er'].reset_index(drop=True)


# In[7]:


dicionarios_reglas = {}
dicionarios_reglas['ATENCION_USUARIO'] = {}
dicionarios_reglas['ATENCION_USUARIO']['servicios'] =['96']
dicionarios_reglas['ATENCION_USUARIO']['motivos'] =['39']
dicionarios_reglas['ATENCION_USUARIO']['canal'] =[]


# In[8]:


dicionarios_reglas


# In[9]:


# Filtramos servicion de interes

# ATENCION AL USUARIO


# In[10]:


data_temp = data_to_process[(data_to_process['Servicio SBS']=='96')&(data_to_process['fecha fase 1'].dt.year>=2019)]
data_temp.head()


# In[11]:


data_temp = data_to_process[(data_to_process['Servicio SBS']=='96')&(data_to_process['Motivo SBS']=='39')]

data_prof = data_to_process[~(data_to_process['Servicio SBS']=='96')]
data_temp.head()


# In[12]:


data_to_process['Servicio BN'].unique()


# In[13]:


# data_temp['Descripcion']


# In[14]:


# Descripcion de hechos ()
100, 101


# In[15]:


data_temp['Descripcion'].values[20]


# # Modelo LDA

# In[16]:


import re

from difflib import SequenceMatcher

import nltk
nltk.download('stopwords')
#from nltk.book import *
from nltk.probability import FreqDist


# In[17]:


# Texto a  minusculas
data_temp['texto']= data_temp['Descripcion'].str.lower()

# nan to  ''
print('Elementos nulos', len(data_temp[data_temp['texto'].isna()]))
data_temp['texto'] = data_temp['texto'].fillna('')
print('Elementos nulos', len(data_temp[data_temp['texto'].isna()]))


# ### Preparacion de palabras clave

# In[18]:


import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop_words = stopwords.words('spanish')
stop_words.extend(["rt", "aun", "oe"])

exclude = string.punctuation
exclude = exclude + "¿"

lemma = SnowballStemmer('spanish')


# In[19]:


# print(stop_words)


# ### Eliminar URL y caracteres de nueva línea

# In[20]:


def remove_three_dots(list_text):
    return [re.sub(r"[a-zA-Z]+(\……|\…)$", " ", texto) for texto in list_text]

def remove_three_dots(list_text):
    return [re.sub("[\.]"*3, " ", texto) for texto in list_text]

def remove_url(list_text):
    return [re.sub(r"http\S+", "", texto).strip() for texto in list_text]

def remove_breakline(list_text):
    return [re.sub('\s+', ' ', texto) for texto in list_text]

def remove_single_quotes(list_text):
    return [re.sub("\'", "", texto) for texto in list_text]


# ### Validando funciones de procesamiento

# In[21]:


txt = data_temp['texto'].values.tolist()
print(txt[20])


# ### Tokenizando palabra y limpieza de texto

# In[22]:


# txt = [data_temp['texto'].values.tolist()[20]+'...']


# In[23]:


import gensim

cleared_txt = remove_url(txt)
cleared_txt = remove_three_dots(cleared_txt)
cleared_txt = remove_breakline(cleared_txt)
cleared_txt = remove_single_quotes(cleared_txt)


# In[24]:


cleared_txt


# In[25]:


def texto_a_palabras(texto:str):
    for sentencia in texto:
        yield(gensim.utils.simple_preprocess(str(sentencia), deacc=True))

def texto_a_palabras2(texto:str):
    for sentencia in texto:
        yield(str(sentencia).split(" "))
        
data_palabras = list(texto_a_palabras2(cleared_txt))


# ### N gramas

# In[26]:


bigram = gensim.models.Phrases(data_palabras, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_palabras], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# ### Eliminar stoprwords, bigramas y lematizando texto

# In[27]:


import spacy
spacy.prefer_gpu()

def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[28]:


data_words_nostops = remove_stopwords(data_palabras)
data_words_bigrams = make_bigrams(data_words_nostops)


# In[29]:


#nlp = spacy.load('es', disable=['parser', 'ner'])
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[30]:


data_lemmatized[20]


# In[31]:


# print(data_words_bigrams[0])


# In[32]:


# print(data_lemmatized[100])


# ### Creacion de diccionario y corpus
# El modelo LDA requiere dos entradas principales: corpus y diccionario

# In[33]:


from gensim import corpora
 
# Diccionario
diccionario = corpora.Dictionary(data_lemmatized)

# Corpus
texto = data_lemmatized
corpus = [diccionario.doc2bow(doc) for doc in texto]


# In[34]:


# print(data_words_bigrams[20])
# print(corpus[20])


# ### Modelo LDA

# In[35]:


# Correr y entrenar el modelo LDA sobre la matriz de términos.
lda_model_ = gensim.models.LdaModel(corpus,
                                   num_topics=4,
                                   id2word = diccionario,
                                   random_state=100,
                                   passes=10,
                                   update_every=1,
                                   chunksize=100,
                                   alpha='auto',
                                   per_word_topics=True)


# In[36]:


import joblib

joblib.dump(lda_model_, 'lda_model.jl')
# then reload it with
lda_model = joblib.load('lda_model.jl')


# In[37]:


lda_model


# ### Lista de temas del modelo LDA

# In[38]:


for idx, topic in lda_model.print_topics():
    print('Topic: {} Word: {}\n'.format(idx, topic))


# In[ ]:





# ### Visualizacion de topicos LDA

# In[39]:


#import warnings
#import pyLDAvis.gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

#warnings.filterwarnings('ignore')

pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model,corpus,diccionario)
vis


# In[40]:


# lda_model.


# In[ ]:





# In[41]:


#                             A B C D
# ATENCION AL USUARIO  [0]    3 4 2 1  = M1 = 4 =
# NO ATENCION AL USUARIO      1 2 1 1  = M2 = 2


# In[42]:


# tfidf = gensim.models.TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]

# corpus_tfidf


# In[43]:


# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=4, id2word=diccionario, passes=10, workers=4)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))


# In[44]:


# Texto para probar - no atencion al ususario


# In[45]:


data_prof['texto']= data_prof['Descripcion'].str.lower()
print('Elementos nulos', len(data_prof[data_prof['texto'].isna()]))

data_prof['texto'] = data_prof['texto'].fillna('')
print('Elementos nulos', len(data_prof[data_prof['texto'].isna()]))

txt_all = data_prof['texto'].values.tolist()

cleared_txt_all = remove_url(txt_all)
cleared_txt_all = remove_three_dots(cleared_txt_all)
cleared_txt_all = remove_breakline(cleared_txt_all)
cleared_txt_all = remove_single_quotes(cleared_txt_all)

data_palabras_all = list(texto_a_palabras2(cleared_txt_all))


# In[46]:


bigram_all = gensim.models.Phrases(data_palabras_all, min_count=5, threshold=100)
trigram_all = gensim.models.Phrases(bigram[data_palabras_all], threshold=100)

bigram_mod_all = gensim.models.phrases.Phraser(bigram_all)
trigram_mod_all = gensim.models.phrases.Phraser(trigram_all)

data_words_nostops_all = remove_stopwords(data_palabras_all)
data_words_bigrams_all = make_bigrams(data_words_nostops_all)


# In[47]:


data_lemmatized_all = lemmatization(data_words_bigrams_all, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

diccionario_all = corpora.Dictionary(data_lemmatized_all)

# Corpus
texto_all = data_lemmatized_all
corpus_all = [diccionario.doc2bow(doc) for doc in texto_all]


# In[48]:


# bow_vector = diccionario.doc2bow(data_palabras_all)


# In[49]:


# corpus[0:10]


# In[50]:


# M1_list = []
# for values, index, score in sorted(lda_model[corpus[0:1000]]):
#     abc = [i[1] for i in values]
#     M1_list.append(np.max(abc))
    
# M2_list = []
# for values, index, score in sorted(lda_model[corpus_all[0:1000]]):
#     abc = [i[1] for i in values]
#     M2_list.append(np.max(abc))
    
# #
# df_results = pd.DataFrame({
#     'Ate. Usuario': M1_list,
#     'Otros': M2_list,
# })
# ax = df_results.plot.kde()


# In[51]:


[0.01907943, 0.07843247, 0.5459396, 0.3565485]


# In[52]:


# M1_list = []
# for values, index, score in sorted(lda_model[corpus[0:1000]]):
#     abc = [i[1] for i in values]
#     M1_list.append(np.max(abc))
    
# M2_list = []
# for values, index, score in sorted(lda_model[corpus_all[0:1000]]):
#     abc = [i[1] for i in values]
#     M2_list.append(np.min(abc))
    
# #
# df_results = pd.DataFrame({
#     'Ate. Usuario': M1_list,
#     'Otros': M2_list,
# })
# ax = df_results.plot.kde()


# In[ ]:





# In[53]:


### TOdos los valores

features = []
target = []
for values, index, score in sorted(lda_model[corpus[0:1000]]):
    abc = [i[1] for i in values]
    features.append(np.array(abc))
    target.append(1)
    
for values, index, score in sorted(lda_model[corpus_all[0:1000]]):
    abc = [i[1] for i in values]
    features.append(np.array(abc))
    target.append(0)
    
df_results = pd.DataFrame(features, columns = ['A', 'B', 'C', 'D'])
df_results['target']=target
df_results = df_results.fillna(0)
df_results.head()


# In[54]:


df_results[df_results['target']==0]


# In[58]:


df_results.to_csv('data/dataset_lda.csv')


# In[55]:


features = np.array(features)
target = np.array(target)

features = df_results[['A', 'B', 'C', 'D']].values
target = df_results['target'].values


# In[57]:


features


# In[ ]:



# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2,random_state=109) 


# In[ ]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
#clf = svm.SVC(kernel='linear', C=0.1) # Linear Kernel
clf = svm.SVC(kernel='poly',degree=3, coef0=1, C=0.1) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

# #Predict the response for test dataset
Y_pred = clf.predict(X_test)


# In[ ]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, Y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[ ]:





# In[63]:


# PRUEBA DEL MODELO svm


# In[64]:


clf


# In[ ]:





# In[65]:


joblib.dump(clf, 'svm_model.jl')
# then reload it with
clf_ = joblib.load('svm_model.jl')


# In[74]:


clf_.predict([[0.047343,0.384974,0.365631,0.202052]])


# In[76]:


abc = [0.055009,0.646546,0.258522,0.039922]
abc = [0.051496,0.3332,0.405048,0.210256]
clf_.predict([abc])


# # Procesos para nuevos Reclamos - Resultados 

# #### Leyendo fuentes

# In[67]:


# Reclamos
file_name = 'data/data_merged.csv'
data_pending_process = pd.read_pickle(file_name)


# #### Filtrando data

# In[68]:


data_pending_process = data_pending_process[(data_pending_process['fecha fase 1'].dt.year>=2021) &
                                            (data_pending_process['fecha fase 1'].dt.month>=7) & 
                                            (data_pending_process['fecha fase 3'].isnull())
                                           ]

#data_pending_process.head()
print('Procesar',len(data_pending_process), 'datos')


# In[69]:



# Filtramos solo nivel 1
data_pending_process = data_pending_process[data_pending_process['Nivel']=='1er'].reset_index(drop=True)

# Fitering to "Atencion al usuario"
data_pending_process = data_pending_process[(data_pending_process['Servicio SBS']=='96')]

#data_pending_process.head()
print('Procesar',len(data_pending_process), 'datos')


# #### Preparacion del texto

# In[70]:


data_pending_process['texto']= data_pending_process['Descripcion'].str.lower()
data_pending_process['texto'] = data_pending_process['texto'].fillna('')

txt_pending = data_pending_process['texto'].values.tolist()
cleared_txt_pending = remove_url(txt_pending)
cleared_txt_pending = remove_three_dots(cleared_txt_pending)
cleared_txt_pending = remove_breakline(cleared_txt_pending)
cleared_txt_pending = remove_single_quotes(cleared_txt_pending)

data_palabras_pending = list(texto_a_palabras2(cleared_txt_pending))

bigram_pending = gensim.models.Phrases(data_palabras_pending, min_count=5, threshold=100)
trigram_pending = gensim.models.Phrases(bigram[data_palabras_pending], threshold=100)

bigram_mod_pending = gensim.models.phrases.Phraser(bigram_pending)
trigram_mod_pending = gensim.models.phrases.Phraser(trigram_pending)

data_words_nostops_pending = remove_stopwords(data_palabras_pending)
data_words_bigrams_pending = make_bigrams(data_words_nostops_pending)

data_lemmatized_pending = lemmatization(data_words_bigrams_pending, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
corpus_pendig = [diccionario.doc2bow(doc) for doc in data_lemmatized_pending]


# In[71]:


worlds_topic = []
umbral = []
for values, index, score in sorted(lda_model[corpus_pendig]):
    # print(values)
    abc = [i[1] for i in values]
    max_val = np.max(abc)
#     umbral.append(max_val)
#     worlds_topic.append(abc.index(max_val)+1)
    
    umbral.append(abc)
    worlds_topic.append(abc.index(max_val)+1)

data_pending_process['worlds_topic'] = worlds_topic
data_pending_process['umbral'] = umbral


# In[ ]:





# In[72]:


data_pending_process[['Numero reclamo', 'Nombre', 'Numero documento',  'Descripcion', 'Departamento', 'Provincia', 'Distrito', 
       'fecha fase 1', 'Canal Incidencia SBS', 'Servicio SBS', 'Motivo SBS',
       'Canal Incidencia', 'Servicio', 'Motivo', 'Nivel', 'texto',
       'worlds_topic', 'umbral']]


# In[ ]:





# In[ ]:




