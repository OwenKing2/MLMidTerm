import csv
import itertools
import warnings

import nltk
import pandas as pd
import sklearn
import spacy
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings(action='ignore')


def simplified_preprocessing(filename):
    header = ["train_id", "Sentence_1", "Sentence_2", "Output"]
    df = pd.read_csv(filename, sep='\t', names=header, engine='python', encoding='utf8', error_bad_lines=False,
                     quoting=csv.QUOTE_NONE)
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

    features["BLEU_1"] = bleu1
    features["BLEU_2"] = bleu2
    features["BLEU_3"] = bleu3
    features["BLEU_4"] = bleu4

    return features


def meteor_scores(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["METEOR"])

    meteor_score = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        meteor_score.append(nltk.translate.meteor_score.single_meteor_score(first_sentence, second_sentence))
    features["METEOR"] = meteor_score
    return features


def character_bigrams_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["CharacterBigramUnion", "CharacterBigramIntersection",
                                     "NumCharBigrams1", "NumCharBigrams2",
                                     ])
    bigramUnion = []
    bigramIntersection = []
    numbigrams1 = []
    numbigrams2 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_no_spaces = sentence_1.replace(" ", "")
        sentence_2_no_spaces = sentence_2.replace(" ", "")
        sentence_1_char_bigrams = [sentence_1_no_spaces[i:i + 2] for i in range(len(sentence_1_no_spaces) - 1)]
        sentence_2_char_bigrams = [sentence_2_no_spaces[i:i + 2] for i in range(len(sentence_2_no_spaces) - 1)]
        bigram_matches = 0
        for phrase in sentence_1_char_bigrams:
            if phrase in sentence_2_char_bigrams:
                bigram_matches += 1
        bigramIntersection.append(bigram_matches)
        bigramUnion.append(len(sentence_1_char_bigrams) + len(sentence_2_char_bigrams))
        numbigrams1.append(len(sentence_1_char_bigrams))
        numbigrams2.append(len(sentence_2_char_bigrams))
    features["CharacterBigramUnion"] = bigramUnion
    features["CharacterBigramIntersection"] = bigramIntersection
    features["NumCharBigrams1"] = numbigrams1
    features["NumCharBigrams2"] = numbigrams2

    return features


def word_unigram_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["SentenceUnigramUnion", "SentenceUnigramIntersection",
                                     "NumSentUnigrams1", "NumSentUnigrams2"])
    unigramUnion = []
    unigramIntersection = []
    numunigrams1 = []
    numunigrams2 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_words = nltk.word_tokenize(sentence_1)
        sentence_2_words = nltk.word_tokenize(sentence_2)
        sentence_1_unigrams = list(nltk.ngrams(sentence_1_words, 1))
        sentence_2_unigrams = list(nltk.ngrams(sentence_2_words, 1))
        unigram_matches = 0
        for phrase in sentence_1_unigrams:
            if phrase in sentence_2_unigrams:
                unigram_matches += 1
        unigramIntersection.append(unigram_matches)
        unigramUnion.append(len(sentence_1_unigrams) + len(sentence_2_unigrams))
        numunigrams1.append(len(sentence_1_unigrams))
        numunigrams2.append(len(sentence_2_unigrams))
    features["SentenceUnigramUnion"] = unigramUnion
    features["SentenceUnigramIntersection"] = unigramIntersection
    features["NumSentUnigrams1"] = numunigrams1
    features["NumSentUnigrams2"] = numunigrams2
    return features


def all_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=[
        "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4",
        "Meteor Score",
        "CharacterBigramUnion", "CharacterBigramIntersection", "NumCharBigrams1", "NumCharBigrams2",
        "SentenceUnigramUnion", "SentenceUnigramIntersection", "NumSentUnigrams1", "NumSentUnigrams2",
    ])
    bleu_scores = bleu_score(sentence1array, sentence2array)

    features["BLEU_1"] = bleu_scores["BLEU_1"]
    features["BLEU_2"] = bleu_scores["BLEU_2"]
    features["BLEU_3"] = bleu_scores["BLEU_3"]
    features["BLEU_4"] = bleu_scores["BLEU_4"]

    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)

    char_bigram = character_bigrams_features(sentence1array, sentence2array)
    word_unigram = word_unigram_features(sentence1array, sentence2array)

    features["CharacterBigramUnion"] = char_bigram["CharacterBigramUnion"]
    features["CharacterBigramIntersection"] = char_bigram["CharacterBigramIntersection"]
    features["NumCharBigrams1"] = char_bigram["NumCharBigrams1"]
    features["NumCharBigrams2"] = char_bigram["NumCharBigrams2"]

    features["SentenceUnigramUnion"] = word_unigram["SentenceUnigramUnion"]
    features["SentenceUnigramIntersection"] = word_unigram["SentenceUnigramIntersection"]
    features["NumSentUnigrams1"] = word_unigram["NumSentUnigrams1"]
    features["NumSentUnigrams2"] = word_unigram["NumSentUnigrams2"]

    return features


training_data = simplified_preprocessing("train_with_label.txt")
X = all_features(training_data["Sentence_1"], training_data["Sentence_2"])
y = training_data["Output"]

dev_data = simplified_preprocessing("dev_with_label.txt")
Xdev = all_features(dev_data["Sentence_1"], dev_data["Sentence_2"])
ydev = dev_data["Output"]

# Code to find the optimal Logistic Regression Model

std_slc = StandardScaler()
min_max = MinMaxScaler()
pca = decomposition.PCA()
logModel = LogisticRegression()
pipe = Pipeline(steps=[('min_max', min_max),
                       ('pca', pca),
                       ('logistic_Reg', logModel)])
n_components = list(range(1, X.shape[1] + 1, 1))
C = [0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 2.5]
# Grid_params = [
#     {'logistic_Reg__solver': ['saga'],
#      'logistic_Reg__penalty': ['elasticnet', 'l1', 'l2', 'none'],
#      'pca__n_components': n_components,
#      'logistic_Reg__C': C},
#     {'logistic_Reg__solver': ['newton-cg'],
#      'logistic_Reg__penalty': ['l2', 'none'],
#      'pca__n_components': n_components,
#      'logistic_Reg__C': C},
#     {'logistic_Reg__solver': ['lbfgs'],
#      'logistic_Reg__penalty': ['l2', 'none'],
#      'pca__n_components': n_components,
#      'logistic_Reg__C': C},
#     {'logistic_Reg__solver': ['liblinear'],
#      'logistic_Reg__penalty': ['l1', 'l2'],
#      'pca__n_components': n_components,
#      'logistic_Reg__C': C},
#     {'logistic_Reg__solver': ['sag'],
#      'logistic_Reg__penalty': ['l2', 'none'],
#      'pca__n_components': n_components,
#      'logistic_Reg__C': C}
# ]
none_newton_params = [{'logistic_Reg__solver': ['newton-cg'],
                       'logistic_Reg__penalty': ['none'],
                       'pca__n_components': n_components,
                       'logistic_Reg__C': C}]

clf_GS = GridSearchCV(pipe, none_newton_params, verbose=2)
clf_GS.fit(X, y)

print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(clf_GS.best_estimator_.get_params()['logistic_Reg'])
print("Optimized logistic regression model accuracy:" + str(clf_GS.score(Xdev, ydev)))


# Code to find the optimal Support Vector Classifier Model
# pipeSVC = Pipeline(steps=[('std_slc', StandardScaler()),
#                                            ('pca', decomposition.PCA()),
#                                            ('svc', SVC())])
#
# kernel = ['linear']
# C = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# # gamma = ['scale', 'auto']
# parametersSVC = dict(pca__n_components=n_components,
#                      svc__kernel=kernel,
#                      svc__C=C,
#                      # svc__gamma=gamma
#                      )
#
# clf_SVC = GridSearchCV(pipeSVC, parametersSVC, verbose=3)
# clf_SVC.fit(X, y)
# print('Best kernel:', clf_SVC.best_estimator_.get_params()['svc__kernel'])
# print('Best C:', clf_SVC.best_estimator_.get_params()['svc__C'])
# # print('Best gamma:', clf_SVC.best_estimator_.get_params()['svc__gamma'])
# print('Best Number Of Components:', clf_SVC.best_estimator_.get_params()['pca__n_components'])
# print(clf_SVC.best_estimator_.get_params()['svc'])
# print("Best SVC model accuracy: " + str(clf_SVC.score(Xdev, ydev)))


def test_preprocess(filename):
    header = ["test_id", "Sentence_1", "Sentence_2"]
    df = pd.read_csv(filename, quoting=3, encoding='utf8', error_bad_lines=False, names=header,
                     sep='\t')
    # Make all words lowercase
    df['Sentence_1'] = df['Sentence_1'].str.lower()
    df['Sentence_2'] = df['Sentence_2'].str.lower()
    return df


test_data = test_preprocess("test_without_label.txt")
Xtest = all_features(test_data["Sentence_1"], test_data["Sentence_2"])
Xoutput = pd.DataFrame()
Xoutput['test_id'] = test_data["test_id"]
Xoutput['prediction'] = clf_GS.predict(Xtest)

Xoutput.to_csv("OwenFitzgeraldKing_test_result.txt", sep='\t', index=False, header=False)
