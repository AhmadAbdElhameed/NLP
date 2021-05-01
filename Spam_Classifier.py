# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:49:22 2021

@author: Ahmad Abd Elhameed
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:14:41 2021

@author: Ahmad Abd Elhameed
"""

import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pandas as pd
import numpy as np

df = pd.read_csv("SMSSpamCollection.txt",
                 sep = '\t',
                 names = ['labels','message'])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []

## Apply Stemming
for i in range(0,len(df)):
    msg = re.sub('[^a-zA-Z]',' ',df["message"][i])
    msg = msg.lower()
    msg = msg.split()
    msg = [stemmer.stem(word) for word in msg if word not in stopwords.words('english')]
    msg = " ".join(msg)
    corpus.append(msg)

## Creating the Bag Of Words model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features= 2000)
X = vectorizer.fit_transform(corpus).toarray()

y = pd.get_dummies(df['labels'])
y = y.iloc[:,1]  ## ham = 1 & spam = 0


## Train & Test Split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

## Training model using Naive bayes classifier 
from sklearn.naive_bayes import MultinomialNB

naive_model = MultinomialNB().fit(X_train,y_train)
y_pred = naive_model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy for Bag-Of-Words Model || Stemming = ',accuracy_score(y_test,y_pred))


## Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

y = pd.get_dummies(df['labels'])
y = y.iloc[:,1]  ## ham = 1 & spam = 0

## Train & Test Split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

## Training model using Naive bayes classifier 
from sklearn.naive_bayes import MultinomialNB

naive_model = MultinomialNB().fit(X_train,y_train)
y_pred = naive_model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy for TF-IDF Model || Stemming = ',accuracy_score(y_test,y_pred))


## ## Apply Lemmatization  ## ##
corpus = []
for i in range(0,len(df)):
    msg = re.sub('[^a-zA-Z]',' ',df["message"][i])
    msg = msg.lower()
    msg = msg.split()
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in stopwords.words('english')]
    msg = " ".join(msg)
    corpus.append(msg)

## Creating the Bag Of Words model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features= 2000)
X = vectorizer.fit_transform(corpus).toarray()

y = pd.get_dummies(df['labels'])
y = y.iloc[:,1]  ## ham = 1 & spam = 0


## Train & Test Split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

## Training model using Naive bayes classifier 
from sklearn.naive_bayes import MultinomialNB

naive_model = MultinomialNB().fit(X_train,y_train)
y_pred = naive_model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy for Bag-Of-Words Model || Lemmatization = ",accuracy_score(y_test,y_pred))

