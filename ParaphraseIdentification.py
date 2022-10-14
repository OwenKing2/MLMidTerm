import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import wordnet
import math
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import codecs
import string
import itertools


def datapreprocess(filename):
    header = ["train_id", "Sentence_1", "Sentence_2", "Output"]
    df = pd.read_csv(filename, sep='\t', on_bad_lines='skip', names=header, engine='python')
    stop_words = set(stopwords.words('english'))
    df['Sentence_1'] = df['Sentence_1'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['Sentence_2'] = df['Sentence_2'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        st = ""
        for w in w_tokenizer.tokenize(text):
            st = st + lemmatizer.lemmatize(w) + " "
        return st

    # gets roots of words for comparison
    df['Sentence_1'] = df['Sentence_1'].apply(lemmatize_text)
    df['Sentence_2'] = df['Sentence_2'].apply(lemmatize_text)

    # Remove punctuation
    df['Sentence_1'] = df['Sentence_1'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['Sentence_2'] = df['Sentence_2'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Make all words lowercase
    df['Sentence_1'] = df['Sentence_1'].str.lower()
    df['Sentence_2'] = df['Sentence_2'].str.lower()

    return df.dropna()


def feature_extractor(sentence1array, sentence2array):
    features = pd.DataFrame(columns=['Length Comparison', 'Proportion of Matching Words', "2grams", "3grams"])

    # Length Dissimilarity: If one is much shorter or longer, it will be a higher value
    length = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        length.append(abs(len(sentence_1) - len(sentence_2)))
    features['Length Comparison'] = length

    # total number of matching words divided by total number of unique words
    matching = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        combined_sentence = sentence_1 + sentence_2
        unique_words = len(set(combined_sentence.split(' ')))
        sentence_1_words = sentence_1.split(" ")
        sentence_2_words = sentence_2.split(" ")
        common_words = len(list(set(sentence_1_words) & set(sentence_2_words)))
        matching.append(common_words / unique_words)
    features['Proportion of Matching Words'] = matching

    # ngrams matches (currently checking 2 and 3 grams)
    twograms = []
    threegrams = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_words = nltk.word_tokenize(sentence_1)
        sentence_2_words = nltk.word_tokenize(sentence_2)
        sentence_1_bigrams = list(nltk.bigrams(sentence_1_words))
        sentence_1_trigrams = list(nltk.trigrams(sentence_1_words))
        sentence_2_bigrams = list(nltk.bigrams(sentence_2_words))
        sentence_2_trigrams = list(nltk.trigrams(sentence_2_words))

        # I will use the total number of bigrams and trigrams in sentence 2 as the bottom of my ratio
        bigramMatches = 0
        for phrase in sentence_1_bigrams:
            if phrase in sentence_2_bigrams:
                bigramMatches += 1
        trigramMatches = 0
        for phrase in sentence_1_trigrams:
            if phrase in sentence_2_trigrams:
                trigramMatches += 1
        twograms.append(bigramMatches / len(sentence_2_bigrams))
        threegrams.append(trigramMatches / len(sentence_2_trigrams))
    features['2grams'] = twograms
    features['3grams'] = threegrams

    # synonyms and hypernyms to be implemented later, want to focus on getting some level of output first

    return features


training_data = datapreprocess("train_with_label.txt")
X = feature_extractor(training_data["Sentence_1"], training_data["Sentence_2"])
y = training_data["Output"]
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)

dev_data = datapreprocess("dev_with_label.txt")
Xdev = feature_extractor(dev_data["Sentence_1"], dev_data["Sentence_2"])
ydev = dev_data["Output"]

print(clf.score(Xdev, ydev))
