import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
import numpy as np

bagOfWords = CountVectorizer()
tfIdfBow = TfidfVectorizer()
data = pd.read_csv("Question2 Dataset.tsv", usecols=['sentiment', 'review'], delimiter='\t',
                   dtype={'sentiment': int, 'review': str})

texts = data['review'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
labels = data['sentiment']
bowVectors = bagOfWords.fit_transform(texts)
tfIdfVectors = tfIdfBow.fit_transform(texts)
traindata, testdata, trainlabel, testlabel = train_test_split(tfIdfVectors, labels, test_size=0.2)
traindata2, testdata2, trainlabel2, testlabel2 = train_test_split(bowVectors, labels, test_size=0.2)
model = MultinomialNB()
model.fit(traindata, trainlabel)
model2 = MultinomialNB()
model2.fit(traindata2, traindata2)
predictions = model.predict(testdata)
predictions2 = model2.predict(testdata2)
print("Accuracy TF-IDF: ", np.mean(predictions == testlabel)*100)
print("Accuracy Bag of Words: ", np.mean(predictions2 == testlabel2)*100)
