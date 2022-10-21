import numpy as np
import pandas as pd
import sklearn.preprocessing
from IPython.display import display
from tabulate import tabulate
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
    # nltk.download('omw-1.4')
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    #
    # try:
    #     nltk.data.find('tokenizers/punkt')
    # except LookupError:
    #     nltk.download('punkt')

    header = ["train_id", "Sentence_1", "Sentence_2", "Output"]
    df = pd.read_csv(filename, sep='\t', on_bad_lines='skip', names=header, engine='python')
    stop_words = set(stopwords.words('english'))
    # Remove stop words like and, a, the
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


def bleu_score(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["BLEU_SCORES"])
    bleu_scores = []
    smooth = SmoothingFunction()
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        bleu_scores.append(round(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                                         smoothing_function=smooth.method3)))
    features["BLEU_SCORES"] = bleu_scores
    return features


def feature_extractor(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["Length Comparison", "Union", "Proportion of Matching Words", "bleu_score"])

    # Length Dissimilarity: If one is much shorter or longer, it will be a higher value
    length = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        length.append(abs(len(sentence_1) - len(sentence_2)) / ((len(sentence_1) + len(sentence_2)) / 2))
    features['Length Comparison'] = length

    # Union: a feature for total number of unique tokens, so that matching words aren't favored too highly
    # for longer sentences, which could naturally have more

    # Matching: total number of matching words, like 1-grams
    matching = []
    union = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        combined_sentence = sentence_1 + sentence_2
        unique_words = len(set(combined_sentence.split(' ')))
        sentence_1_words = sentence_1.split(" ")
        sentence_2_words = sentence_2.split(" ")
        common_words = len(list(set(sentence_1_words) & set(sentence_2_words)))
        matching.append(common_words)
        union.append(unique_words)
    features['Proportion of Matching Words'] = matching
    features['Union'] = union
    # ngrams matches (currently checking 2 and 3 grams)
    # unigrams = []
    # twograms = []
    # threegrams = []
    # fourgrams = []
    # for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
    #     sentence_1_words = nltk.word_tokenize(sentence_1)
    #     sentence_2_words = nltk.word_tokenize(sentence_2)
    #     sentence_1_unigrams = list(nltk.ngrams(sentence_1_words, 1))
    #     sentence_2_unigrams = list(nltk.ngrams(sentence_1_words, 1))
    #     sentence_1_bigrams = list(nltk.bigrams(sentence_1_words))
    #     sentence_1_trigrams = list(nltk.trigrams(sentence_1_words))
    #     sentence_2_bigrams = list(nltk.bigrams(sentence_2_words))
    #     sentence_2_trigrams = list(nltk.trigrams(sentence_2_words))
    #     sentence_1_fourgrams = list(nltk.ngrams(sentence_1_words, 4))
    #     sentence_2_fourgrams = list(nltk.ngrams(sentence_1_words, 4))
    #
    #     # I will use the total number of bigrams and trigrams in sentence 2 as the bottom of my ratio
    #     unigram_matches = 0
    #     for phrase in sentence_1_unigrams:
    #         if phrase in sentence_2_unigrams:
    #             unigram_matches += 1
    #
    #     bigram_matches = 0
    #     for phrase in sentence_1_bigrams:
    #         if phrase in sentence_2_bigrams:
    #             bigram_matches += 1
    #
    #     trigram_matches = 0
    #     for phrase in sentence_1_trigrams:
    #         if phrase in sentence_2_trigrams:
    #             trigram_matches += 1
    #
    #     fourgram_matches = 0
    #     for phrase in sentence_1_fourgrams:
    #         if phrase in sentence_2_fourgrams:
    #             fourgram_matches += 1
    #
    #     unigrams.append(unigram_matches)
    #     twograms.append(bigram_matches)
    #     threegrams.append(trigram_matches)
    #     fourgrams.append(fourgram_matches)

    # features['1grams'] = unigrams
    # features['2grams'] = twograms
    # features['3grams'] = threegrams
    # features['4grams'] = fourgrams

    features["bleu_score"] = bleu_score(sentence1array, sentence2array)
    # synonyms and hypernyms to be implemented later, want to focus on getting some level of output first

    return features


def vectorize_features(sentence1array, sentence2array):
    return []


training_data = datapreprocess("train_with_label.txt")
X = feature_extractor(training_data["Sentence_1"], training_data["Sentence_2"])
y = training_data["Output"]

dev_data = datapreprocess("dev_with_label.txt")
Xdev = feature_extractor(dev_data["Sentence_1"], dev_data["Sentence_2"])
ydev = dev_data["Output"]

SVC = make_pipeline(StandardScaler(), SVC(kernel="sigmoid"))
SVC.fit(X, y)

linearSVC = make_pipeline(StandardScaler(), LinearSVC(dual=False, tol=1e-5))
linearSVC.fit(X, y)

logisticRegression = make_pipeline(StandardScaler(), LogisticRegression())
logisticRegression.fit(X, y)

print("SVC model accuracy: " + str(SVC.score(Xdev, ydev)))
print("linearSVC model accuracy: " + str(linearSVC.score(Xdev, ydev)))
print("Logistic Regression model accuracy: " + str(logisticRegression.score(Xdev, ydev)))
