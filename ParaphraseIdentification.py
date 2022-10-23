import itertools
import string

import nltk
import numpy as np
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sklearn import linear_model, decomposition
import en_core_web_sm
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings(action='ignore')


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

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        first_sentence = nltk.word_tokenize(sentence_1)
        second_sentence = nltk.word_tokenize(sentence_2)
        nist1.append(nltk.translate.nist_score.sentence_nist([first_sentence], second_sentence, n=1))
        nist2.append(nltk.translate.nist_score.sentence_nist([first_sentence], second_sentence, n=2))
        nist3.append(nltk.translate.nist_score.sentence_nist([first_sentence], second_sentence, n=3))
        nist4.append(nltk.translate.nist_score.sentence_nist([first_sentence], second_sentence, n=4))
        nist5.append(nltk.translate.nist_score.sentence_nist([first_sentence], second_sentence, n=5))

    features["NIST_1"] = nist1
    features["NIST_2"] = nist2
    features["NIST_3"] = nist3
    features["NIST_4"] = nist4
    features["NIST_5"] = nist5

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


# Currently my highest accuracy (with NIST features removed acc ~64%
def all_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=[
    #    "NIST_1", "NIST_2", "NIST_3", "NIST_4", "NIST_5",
        "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4",
        "Cosine Similarity",
        "Meteor Score",
        "Character Bigram Union", "Character Bigram Intersection", "NumBigrams1", "NumBigrams2",
        "Sentence Unigram Union", "Sentence Unigram Intersection", "NumUnigrams1", "NumUnigrams2"
    ])
    nist_scores = nist_score(sentence1array, sentence2array)
    bleu_scores = bleu_score(sentence1array, sentence2array)

    features["BLEU_1"] = bleu_scores["BLEU_1"]
    features["BLEU_2"] = bleu_scores["BLEU_2"]
    features["BLEU_3"] = bleu_scores["BLEU_3"]
    features["BLEU_4"] = bleu_scores["BLEU_4"]

    # features["NIST_1"] = nist_scores["NIST_1"]
    # features["NIST_2"] = nist_scores["NIST_2"]
    # features["NIST_3"] = nist_scores["NIST_3"]
    # features["NIST_4"] = nist_scores["NIST_4"]
    # features["NIST_5"] = nist_scores["NIST_5"]

    features["Cosine Similarity"] = vectorize_features(sentence1array, sentence2array)
    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)

    charBigramFeatures = character_bigrams_features(sentence1array, sentence2array)
    features["Character Bigram Union"] = charBigramFeatures["Character Bigram Union"]
    features["Character Bigram Intersection"] = charBigramFeatures["Character Bigram Intersection"]
    features["NumBigrams1"] = charBigramFeatures["NumBigrams1"]
    features["NumBigrams2"] = charBigramFeatures["NumBigrams2"]

    wordUnigramFeatures = word_unigram_features(sentence1array, sentence2array)
    features["Sentence Unigram Union"] = wordUnigramFeatures["Sentence Unigram Union"]
    features["Sentence Unigram Intersection"] = wordUnigramFeatures["Sentence Unigram Intersection"]
    features["NumUnigrams1"] = wordUnigramFeatures["NumUnigrams1"]
    features["NumUnigrams2"] = wordUnigramFeatures["NumUnigrams2"]

    return features


def feature_extractor(sentence1array, sentence2array):
    features = pd.DataFrame(
        columns=["Length Comparison", "Union", "Proportion of Matching Words",
                 "METEOR"])

    # Length Dissimilarity: If one is much shorter or longer, it will be a higher value
    length = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        length.append(abs(len(sentence_1) - len(sentence_2)) / ((len(sentence_1) + len(sentence_2)) / 2))
    features['Length Comparison'] = length
    # length_1 = []
    # length_2 = []
    # for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
    #     length_1.append(len(sentence_1))
    #     length_2.append(len(sentence_2))
    # features["Length One"] = length_1
    # features["Length Two"] = length_2
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

    features["METEOR"] = meteor_scores(sentence1array, sentence2array)
    # nist_scores = nist_score(sentence1array, sentence2array)
    # features["NIST_1"] = nist_scores["NIST_1"]
    # features["NIST_2"] = nist_scores["NIST_2"]
    # features["NIST_3"] = nist_scores["NIST_3"]
    # features["NIST_4"] = nist_scores["NIST_4"]
    # features["NIST_5"] = nist_scores["NIST_5"]

    # synonyms and hypernyms to be implemented later, want to focus on getting some level of output first

    return features


def vectorize_features(sentence1array, sentence2array):
    # features = pd.DataFrame(columns=["Word Union", "Word Intersection", "Word 1 Length", "Word 2 Length",
    #                                 "Bigram Union", "Bigram Intersection", "Bigram 1 Length", "Bigram 2 Length"])

    features = pd.DataFrame(columns=["Sentence Cosine Similarity"])
    # sentence1list = []
    # sentence2list = []
    sentence1vectors = []
    sentence2vectors = []
    embeddings = spacy.load('en_core_web_sm')

    for sentence in sentence1array:
        # sentence1list.append(sentence)
        sentence1vectors.append(embeddings(sentence).vector)
    for sentence in sentence2array:
        # sentence2list.append(sentence)
        sentence2vectors.append(embeddings(sentence).vector)
    cosinesim = []
    for (vector_1, vector_2) in itertools.zip_longest(sentence1vectors, sentence2vectors):
        cosinesim.append(cosine_similarity([vector_1], [vector_2]))
    features["Sentence Cosine Similarity"] = cosinesim

    return features


def meteor_and_vector(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["Sentence Cosine Similarity", "Meteor Score", "Length Comparison", "Union",
                                     "Proportion of Matching Words"])
    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)
    features["Sentence Cosine Similarity"] = vectorize_features(sentence1array, sentence2array)
    length = []
    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        length.append(abs(len(sentence_1) - len(sentence_2)) / ((len(sentence_1) + len(sentence_2)) / 2))
    features['Length Comparison'] = length
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
    return features


def character_bigrams_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["Character Bigram Union", "Character Bigram Intersection",
                                     "NumBigrams1", "NumBigrams2"])
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
    features["Character Bigram Union"] = bigramUnion
    features["Character Bigram Intersection"] = bigramIntersection
    features["NumBigrams1"] = numbigrams1
    features["NumBigrams2"] = numbigrams2

    return features


def word_unigram_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["Sentence Unigram Union", "Sentence Unigram Intersection",
                                     "NumUnigrams1", "NumUnigrams2"])
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
    features["Sentence Unigram Union"] = unigramUnion
    features["Sentence Unigram Intersection"] = unigramIntersection
    features["NumUnigrams1"] = numunigrams1
    features["NumUnigrams2"] = numunigrams2
    return features


def charBigramWordUnigram(sentence1array, sentence2array):
    features = pd.DataFrame(columns=[
        # "Sentence Unigram Union", "Sentence Unigram Intersection", "NumUnigrams1", "NumUnigrams2",
        # "Character Bigram Union", "Character Bigram Intersection", "NumBigrams1", "NumBigrams2",
        "Meteor Score",
        "Cosine Similarity"
    ])
    charBigramFeatures = character_bigrams_features(sentence1array,sentence2array)
    # wordUnigramFeatures = word_unigram_features(sentence1array,sentence2array)

    # features["Character Bigram Union"] = charBigramFeatures["Character Bigram Union"]
    # features["Character Bigram Intersection"] = charBigramFeatures["Character Bigram Intersection"]
    # features["NumBigrams1"] = charBigramFeatures["NumBigrams1"]
    # features["NumBigrams2"] = charBigramFeatures["NumBigrams2"]

    # features["Sentence Unigram Union"] = wordUnigramFeatures["Sentence Unigram Union"]
    # features["Sentence Unigram Intersection"] = wordUnigramFeatures["Sentence Unigram Intersection"]
    # features["NumUnigrams1"] = wordUnigramFeatures["NumUnigrams1"]
    # features["NumUnigrams2"] = wordUnigramFeatures["NumUnigrams2"]
    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)
    features["Cosine Similarity"] = vectorize_features(sentence1array, sentence2array)

    return features


training_data = simplified_preprocessing("train_with_label.txt")
X = all_features(training_data["Sentence_1"], training_data["Sentence_2"])
y = training_data["Output"]

dev_data = simplified_preprocessing("dev_with_label.txt")
Xdev = all_features(dev_data["Sentence_1"], dev_data["Sentence_2"])
ydev = dev_data["Output"]

# Code to find the optimal Logistic Regression Model

std_slc = StandardScaler()
pca = decomposition.PCA()
logModel = LogisticRegression()
pipe = sklearn.pipeline.Pipeline(steps=[('std_slc', std_slc),
                                        ('pca', pca),
                                        ('logistic_Reg', logModel)])
n_components = list(range(1, X.shape[1] + 1, 1))
penalty = 'none',
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
parameters = dict(pca__n_components=n_components,
                  logistic_Reg__solver=solver,
                  logistic_Reg__penalty=penalty)

clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X, y)

print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(clf_GS.best_estimator_.get_params()['logistic_Reg'])
print("Optimized logistic regression model accuracy:" + str(clf_GS.score(Xdev, ydev)))

nonelogisticRegression = make_pipeline(StandardScaler(),
                                       LogisticRegression(penalty="none", solver="newton-cg"))
nonelogisticRegression.fit(X, y)

print("None Logistic Regression Model Accuracy:" + str(nonelogisticRegression.score(Xdev, ydev)))
