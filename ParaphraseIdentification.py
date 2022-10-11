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

# test_header = ["test_id", "Sentence_1", "Sentence_2"]
# test = pd.read_csv('dev_with_label.txt', sep='\t', on_bad_lines='skip', names=header, engine='python')
# test['Sentence_1'] = df['Sentence_1'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
# test['Sentence_2'] = df['Sentence_2'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#
# test['Sentence_1'] = df['Sentence_1'].apply(lemmatize_text)
# test['Sentence_2'] = df['Sentence_2'].apply(lemmatize_text)
#
# removedNull = df.dropna()
# Xtest = removedNull[["Sentence_1", "Sentence_2"]]

# Now, we need to define a corpus.
# I will use all the sentence pairs that appear in the training set

vec = CountVectorizer()
X = vec.fit_transform(X)
vocab = vec.get_feature_names_out()
X = X.toarray()
word_counts = {}
for l in range(2):
    word_counts[l] = defaultdict(lambda: 0)
for i in range(X.shape[0]):
    l = y[i]
    for j in range(len(vocab)):
        word_counts[l][vocab[j]] += X[i][j]


def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a / b)

def group_by_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data


def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors


def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result


labels = [0,1]
n_label_items, log_label_priors = fit(X,y,labels)
pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, Xdev)
print("Accuracy of prediction on test set : ", accuracy_score(ydev,pred))