import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import wordnet
import math
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import codecs

inputFileLineCount = 4077
header = ["train_id", "Sentence_1", "Sentence_2", "Output"]

df = pd.read_csv('train_with_label.txt', sep='\t', on_bad_lines='skip', names=header, engine='python')

# Preprocessing
# nltk.download('omw-1.4')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['Sentence_1'] = df['Sentence_1'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df['Sentence_2'] = df['Sentence_2'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st


df['Sentence_1'] = df['Sentence_1'].apply(lemmatize_text)
df['Sentence_2'] = df['Sentence_2'].apply(lemmatize_text)

removedNull = df.dropna()
X = removedNull[["Sentence_1", "Sentence_2"]]
y = removedNull["Output"]

dev = pd.read_csv('dev_with_label.txt', sep='\t', on_bad_lines='skip', names=header, engine='python')
dev['Sentence_1'] = df['Sentence_1'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
dev['Sentence_2'] = df['Sentence_2'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

dev['Sentence_1'] = df['Sentence_1'].apply(lemmatize_text)
dev['Sentence_2'] = df['Sentence_2'].apply(lemmatize_text)

removedNull = df.dropna()
Xdev = removedNull[["Sentence_1", "Sentence_2"]]
ydev = removedNull["Output"]

test_header = ["test_id", "Sentence_1", "Sentence_2"]
test = pd.read_csv('dev_with_label.txt', sep='\t', on_bad_lines='skip', names=header, engine='python')
test['Sentence_1'] = df['Sentence_1'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
test['Sentence_2'] = df['Sentence_2'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

test['Sentence_1'] = df['Sentence_1'].apply(lemmatize_text)
test['Sentence_2'] = df['Sentence_2'].apply(lemmatize_text)

removedNull = df.dropna()
Xtest = removedNull[["Sentence_1", "Sentence_2"]]

# Now, we need to define a corpus.
# I will use all the words that appear in all datasets, including dev train and test

