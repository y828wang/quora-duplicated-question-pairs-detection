print ('Importing...')
import csv
import re
import gensim
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import scipy
from scipy import sparse
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD


def tokenize(text):
    return text.lower().split()

# OPEN TRAINING DATA SET
print ('Opening Training Data Set...')
with open('train_balanced.csv', 'r') as csvfile:
    writer = csv.reader(csvfile)
    train_sentences = list(writer)
# OPEN TESTING DATA SET
with open('test.csv', 'r') as csvfile:
    writer = csv.reader(csvfile)
    test_sentences = list(writer)

# OVERALL
# stops = stopwords.words("english")
# rdict = {'</span>': '', '<br/>': '', '<br />': ''}
# total_phrases = []  # For shuffling purpose
# FOR TRAINING DATA
tags = []
train_originals = []  # For keeping the structure of the sentence
# train_corpus_set_1 = []  # For keeping the order of the text
# train_corpus_set_2 = []
# FOR TESTING DATA
test_originals = []  # For keeping the structure of the sentence
# test_corpus_set_1 = []  # For keeping the order of the text
# test_corpus_set_2 = []

# PROCESS TRAINING DATA
print ('Processing Training Data...')
for i in range(1, len(train_sentences)):
    input = train_sentences[i]
    set1 = input[3]
    #set1 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set1)  # Deletes all punctuations
    #set1 = set1.lower()
    set2 = input[4]
    #set2 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set2)
    #set2 = set2.lower()
    train_originals.append(set1+set2)
    #train_originals.append(' '.join(set2))
    #train_corpus_set_1.append(set1)
    #train_corpus_set_2.append(set2)
    #total_phrases.append(set1+set2)
    tags.append(input[5])

# PROCESS TESTING DATA
print ('Processing Testing Data...')
for i in range(1, len(test_sentences)):
    input = test_sentences[i]
    set1 = input[1]
    #set1 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set1)  # Deletes all punctuations
    #set1 = set1.lower().split()
    set2 = input[2]
    #set2 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set2)
    #set2 = set2.lower().split()
    test_originals.append(set1+set2)
    #test_originals.append(' '.join(set2))
    #test_corpus_set_1.append(set1)
    #test_corpus_set_2.append(set2)
    #total_phrases.append(set1+set2)

# TF-IDF REPRESENTATION
print ('Generating TF-IDF Representation...')
tfidf = TV(min_df=2, analyzer='word', strip_accents='unicode', tokenizer=tokenize, use_idf=False)
originals = train_originals + test_originals
tfidf.fit(originals) # This is the slow part!
originals = tfidf.transform(originals)

# DIMENSIONALITY REDUCTION - SVD
print ('Reducing dimensions...')
svd = TruncatedSVD(n_components=500, n_iter=8)
svd.fit(originals)
originals = svd.transform(originals)
explained_variance = svd.explained_variance_ratio_.sum()
print ('Explained Variance: ' + str(explained_variance))

# LOGISTIC REGRESSION
print ('Fitting Logistic Regression...')
#customWeights = {'0':1, '1':0.873}
model = LogisticRegression(C=10)
model.fit(originals[:len(train_originals)], tags)
score = model.score(originals[:len(train_originals)], tags)
print ('Accuracy: ' + str(score))
predictions = model.predict_proba(originals[len(train_originals):])
predictions_train = model.predict_proba(originals[:len(train_originals)])

# WRITE RESULTS TO CSV
print ('Writing to CSV...')
good_predict = predictions[:, 1]
good_train = predictions_train[:, 1]
with open('training_fit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["test_id", "is_duplicate"])
    for i in range(0, len(good_train)):
        writer.writerow([i, good_train[i]])
    csvfile.close()
with open('predictions.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["test_id", "is_duplicate"])
    for i in range(0, len(good_predict)):
        writer.writerow([i, good_predict[i]])
    csvfile.close()

print ('COMPLETE')
