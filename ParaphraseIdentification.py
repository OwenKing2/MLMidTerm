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


def simplified_preprocessing(filename):
    header = ["train_id", "Sentence_1", "Sentence_2", "Output"]
    df = pd.read_csv(filename, sep='\t', on_bad_lines='skip', names=header, engine='python')
    # Make all words lowercase
    df['Sentence_1'] = df['Sentence_1'].str.lower()
    df['Sentence_2'] = df['Sentence_2'].str.lower()

    return df.dropna()


def bleu_score(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"])
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    # bleu5 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        bleu1.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[1]))
        bleu2.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[0.5, 0.5]))
        bleu3.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[1 / 3, 1 / 3, 1 / 3]))
        bleu4.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             weights=[1 / 4, 1 / 4, 1 / 4, 1 / 4]))
    # bleu5.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
    #                                                     weights=[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))

    features["BLEU_1"] = bleu1
    features["BLEU_2"] = bleu2
    features["BLEU_3"] = bleu3
    features["BLEU_4"] = bleu4
    # features["BLEU_5"] = bleu1

    return features


def nist_score(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["NIST_1", "NIST_2", "NIST_3", "NIST_4", "NIST_5"])
    nist1 = []
    nist2 = []
    nist3 = []
    nist4 = []
    nist5 = []

    smooth = SmoothingFunction()
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        nist1.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             smoothing_function=smooth.method3,
                                                             weights=[1]))
        nist2.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             smoothing_function=smooth.method3,
                                                             weights=[0.5, 0.5]))
        nist3.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             smoothing_function=smooth.method3,
                                                             weights=[1 / 3, 1 / 3, 1 / 3]))
        nist4.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             smoothing_function=smooth.method3,
                                                             weights=[1 / 4, 1 / 4, 1 / 4, 1 / 4]))
        nist5.append(nltk.translate.bleu_score.sentence_bleu([first_sentence], second_sentence,
                                                             smoothing_function=smooth.method3,
                                                             weights=[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))

    features["NIST_1"] = nist1
    features["NIST_2"] = nist2
    features["NIST_3"] = nist3
    features["NIST_4"] = nist4
    features["NIST_5"] = nist5

    return features


def bleu_nist(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["NIST_1", "NIST_2", "NIST_3", "NIST_4", "NIST_5",
                                     "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"])
    nist_scores = nist_score(sentence1array, sentence2array)
    bleu_scores = bleu_score(sentence1array, sentence2array)

    features["BLEU_1"] = bleu_scores["BLEU_1"]
    features["BLEU_2"] = bleu_scores["BLEU_2"]
    features["BLEU_3"] = bleu_scores["BLEU_3"]
    features["BLEU_4"] = bleu_scores["BLEU_4"]

    features["NIST_1"] = nist_scores["NIST_1"]
    features["NIST_2"] = nist_scores["NIST_2"]
    features["NIST_3"] = nist_scores["NIST_3"]
    features["NIST_4"] = nist_scores["NIST_4"]
    features["NIST_5"] = nist_scores["NIST_5"]
    return features


def feature_extractor(sentence1array, sentence2array):
    features = pd.DataFrame(
        columns=["Length Comparison", "Union", "Proportion of Matching Words",
                 "NIST_1", "NIST_2", "NIST_3", "NIST_4", "NIST_5"])

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

    nist_scores = nist_score(sentence1array, sentence2array)
    features["NIST_1"] = nist_scores["NIST_1"]
    features["NIST_2"] = nist_scores["NIST_2"]
    features["NIST_3"] = nist_scores["NIST_3"]
    features["NIST_4"] = nist_scores["NIST_4"]
    features["NIST_5"] = nist_scores["NIST_5"]

    # synonyms and hypernyms to be implemented later, want to focus on getting some level of output first

    return features


def vectorize_features(sentence1array, sentence2array):
    return []


training_data = simplified_preprocessing("train_with_label.txt")
X = bleu_nist(training_data["Sentence_1"], training_data["Sentence_2"])
y = training_data["Output"]

dev_data = simplified_preprocessing("dev_with_label.txt")
Xdev = bleu_nist(dev_data["Sentence_1"], dev_data["Sentence_2"])
ydev = dev_data["Output"]

# SVC = make_pipeline(StandardScaler(), SVC(kernel="sigmoid"))
# SVC.fit(X, y)
#
# # linearSVC = make_pipeline(StandardScaler(), LinearSVC(dual=False, tol=1e-5))
# # linearSVC.fit(X, y)
#
# logisticRegression = make_pipeline(StandardScaler(), LogisticRegression())
# logisticRegression.fit(X, y)
#
# print("SVC model accuracy: " + str(SVC.score(Xdev, ydev)))
# # print("linearSVC model accuracy: " + str(linearSVC.score(Xdev, ydev)))
# print("Logistic Regression model accuracy: " + str(logisticRegression.score(Xdev, ydev)))


# Will create multiple different SVC and Logistic Regression models to see the effect of tuning parameters

linearSVC = make_pipeline(StandardScaler(), SVC(kernel="linear"))
linearSVC.fit(X, y)

polySVC = make_pipeline(StandardScaler(), SVC(kernel="poly"))
polySVC.fit(X, y)

rbfSVC = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
rbfSVC.fit(X, y)

sigmoidSVC = make_pipeline(StandardScaler(), SVC(kernel="sigmoid"))
sigmoidSVC.fit(X, y)

print("SVC linear model accuracy: " + str(linearSVC.score(Xdev, ydev)))
print("SVC poly model accuracy: " + str(polySVC.score(Xdev, ydev)))
print("SVC rbf model accuracy: " + str(rbfSVC.score(Xdev, ydev)))
print("SVC sigmoid model accuracy: " + str(sigmoidSVC.score(Xdev, ydev)))

l1logisticRegression = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear"))
l1logisticRegression.fit(X, y)

l2logisticRegression = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear"))
l2logisticRegression.fit(X, y)

# elasticlogisticRegression = make_pipeline(StandardScaler(),
# LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5))
# elasticlogisticRegression.fit(X, y)

nonelogisticRegression = make_pipeline(StandardScaler(), LogisticRegression(penalty="none", solver="newton-cg"))
nonelogisticRegression.fit(X, y)

print("Logistic Regression l1 model accuracy: " + str(l1logisticRegression.score(Xdev, ydev)))
print("Logistic Regression l2 model accuracy: " + str(l2logisticRegression.score(Xdev, ydev)))
# print("Logistic Regression elasticnet model accuracy: " + str(elasticlogisticRegression.score(Xdev, ydev)))
print("Logistic Regression none model accuracy: " + str(nonelogisticRegression.score(Xdev, ydev)))
