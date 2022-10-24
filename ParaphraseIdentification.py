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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from itertools import product

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


# Currently my highest accuracy (with NIST features removed acc ~67%
def all_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=[
        "NIST_1",
        "NIST_2",
        "NIST_3",
        "NIST_4",
        "NIST_5",
        "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4",
        "Cosine Similarity",
        "Meteor Score",
        "CharacterBigramUnion", "CharacterBigramIntersection", "NumCharBigrams1", "NumCharBigrams2",
        "SentenceUnigramUnion", "SentenceUnigramIntersection", "NumSentUnigrams1", "NumSentUnigrams2",
        "CharacterUnigramUnion", "CharacterUnigramIntersection", "NumCharUnigrams1", "NumCharUnigrams2",
        "SentenceBigramUnion", "SentenceBigramIntersection", "NumSentBigrams1", "NumSentBigrams2"
    ])
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

    features["Cosine Similarity"] = vectorize_features(sentence1array, sentence2array)
    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)

    charBigramFeatures = character_bigrams_features(sentence1array, sentence2array)
    features["CharacterBigramUnion"] = charBigramFeatures["CharacterBigramUnion"]
    features["CharacterBigramIntersection"] = charBigramFeatures["CharacterBigramIntersection"]
    features["NumCharBigrams1"] = charBigramFeatures["NumCharBigrams1"]
    features["NumCharBigrams2"] = charBigramFeatures["NumCharBigrams2"]

    wordUnigramFeatures = word_unigram_features(sentence1array, sentence2array)
    features["SentenceUnigramUnion"] = wordUnigramFeatures["SentenceUnigramUnion"]
    features["SentenceUnigramIntersection"] = wordUnigramFeatures["SentenceUnigramIntersection"]
    features["NumSentUnigrams1"] = wordUnigramFeatures["NumSentUnigrams1"]
    features["NumSentUnigrams2"] = wordUnigramFeatures["NumSentUnigrams2"]

    char_bigram = character_bigrams_features(sentence1array, sentence2array)
    char_unigram = character_unigrams_features(sentence1array, sentence2array)
    word_unigram = word_unigram_features(sentence1array, sentence2array)
    word_bigram = word_bigram_features(sentence1array, sentence2array)

    # "CharacterBigramUnion", "CharacterBigramIntersection", "NumCharBigrams1", "NumCharBigrams2"
    features["CharacterBigramUnion"] = char_bigram["CharacterBigramUnion"]
    features["CharacterBigramIntersection"] = char_bigram["CharacterBigramIntersection"]
    features["NumCharBigrams1"] = char_bigram["NumCharBigrams1"]
    features["NumCharBigrams2"] = char_bigram["NumCharBigrams2"]

    # "CharacterUnigramUnion", "CharacterUnigramIntersection",
    #                                      "NumCharUnigrams1", "NumCharUnigrams2"
    features["CharacterUnigramUnion"] = char_unigram["CharacterUnigramUnion"]
    features["CharacterUnigramIntersection"] = char_unigram["CharacterUnigramIntersection"]
    features["NumCharUnigrams1"] = char_unigram["NumCharUnigrams1"]
    features["NumCharUnigrams2"] = char_unigram["NumCharUnigrams2"]

    # "SentenceUnigramUnion", "SentenceUnigramIntersection",
    # "NumSentUnigrams1", "NumSentUnigrams2"
    features["SentenceUnigramUnion"] = word_unigram["SentenceUnigramUnion"]
    features["SentenceUnigramIntersection"] = word_unigram["SentenceUnigramIntersection"]
    features["NumSentUnigrams1"] = word_unigram["NumSentUnigrams1"]
    features["NumSentUnigrams2"] = word_unigram["NumSentUnigrams2"]
    # "SentenceBigramUnion", "SentenceBigramIntersection",
    #                                      "NumSentBigrams1", "NumSentBigrams2"
    features["SentenceBigramUnion"] = word_bigram["SentenceBigramUnion"]
    features["SentenceBigramIntersection"] = word_bigram["SentenceBigramIntersection"]
    features["NumSentBigrams1"] = word_bigram["NumSentBigrams1"]
    features["NumSentBigrams2"] = word_bigram["NumSentBigrams2"]

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
    features = pd.DataFrame(columns=["Sentence Cosine Similarity"])
    sentence1vectors = []
    sentence2vectors = []
    embeddings = spacy.load('en_core_web_sm')
    for sentence in sentence1array:
        sentence1vectors.append(embeddings(sentence).vector)
    for sentence in sentence2array:
        sentence2vectors.append(embeddings(sentence).vector)
    cosinesim = []
    for (vector_1, vector_2) in itertools.zip_longest(sentence1vectors, sentence2vectors):
        cosinesim.append(cosine_similarity([vector_1], [vector_2]))
    features["Sentence Cosine Similarity"] = cosinesim

    return features


def meteor_and_vector(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["Sentence Cosine Similarity", "Meteor Score"])
    features["Meteor Score"] = meteor_scores(sentence1array, sentence2array)
    features["Sentence Cosine Similarity"] = vectorize_features(sentence1array, sentence2array)
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


def character_unigrams_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["CharacterUnigramUnion", "CharacterUnigramIntersection",
                                     "NumCharUnigrams1", "NumCharUnigrams2"])
    CharacterUnigramUnion = []
    CharacterUnigramIntersection = []
    NumUnigrams1 = []
    NumUnigrams2 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_no_spaces = sentence_1.replace(" ", "")
        sentence_2_no_spaces = sentence_2.replace(" ", "")
        sentence_1_char_unigrams = list(sentence_1_no_spaces)
        sentence_2_char_unigrams = list(sentence_2_no_spaces)
        unigram_matches = 0
        for char in sentence_1_char_unigrams:
            if char in sentence_2_char_unigrams:
                unigram_matches += 1
        CharacterUnigramIntersection.append(unigram_matches)
        CharacterUnigramUnion.append(len(sentence_1_char_unigrams) + len(sentence_2_char_unigrams))
        NumUnigrams1.append(len(sentence_1_char_unigrams))
        NumUnigrams2.append(len(sentence_2_char_unigrams))
    features["CharacterUnigramUnion"] = CharacterUnigramUnion
    features["CharacterUnigramIntersection"] = CharacterUnigramIntersection
    features["NumCharUnigrams1"] = NumUnigrams1
    features["NumCharUnigrams2"] = NumUnigrams2

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


def word_bigram_features(sentence1array, sentence2array):
    features = pd.DataFrame(columns=["SentenceBigramUnion", "SentenceBigramIntersection",
                                     "NumSentBigrams1", "NumSentBigrams2"])
    SentencebigramUnion = []
    SentencebigramIntersection = []
    Numbigrams1 = []
    Numbigrams2 = []

    for (sentence_1, sentence_2) in itertools.zip_longest(sentence1array, sentence2array):
        sentence_1_words = nltk.word_tokenize(sentence_1)
        sentence_2_words = nltk.word_tokenize(sentence_2)
        sentence_1_bigrams = list(nltk.ngrams(sentence_1_words, 2))
        sentence_2_bigrams = list(nltk.ngrams(sentence_2_words, 2))
        bigram_matches = 0
        for phrase in sentence_1_bigrams:
            if phrase in sentence_2_bigrams:
                bigram_matches += 1
        SentencebigramIntersection.append(bigram_matches)
        SentencebigramUnion.append(len(sentence_1_bigrams) + len(sentence_2_bigrams))
        Numbigrams1.append(len(sentence_1_bigrams))
        Numbigrams2.append(len(sentence_2_bigrams))
    features["SentenceBigramUnion"] = SentencebigramUnion
    features["SentenceBigramIntersection"] = SentencebigramIntersection
    features["NumSentBigrams1"] = Numbigrams1
    features["NumSentBigrams2"] = Numbigrams2
    return features


def word_and_sentence_features(sentence1array, sentence2array):
    features = pd.DataFrame(
        # columns=[
        #     "CharacterBigramUnion", "CharacterBigramIntersection", "NumCharBigrams1", "NumCharBigrams2",
        #     "CharacterUnigramUnion", "CharacterUnigramIntersection", "NumCharUnigrams1", "NumCharUnigrams2",
        #     "SentenceUnigramUnion", "SentenceUnigramIntersection", "NumSentUnigrams1", "NumSentUnigrams2",
        #     "SentenceBigramUnion", "SentenceBigramIntersection", "NumSentBigrams1", "NumSentBigrams2"
        # ]
    )
    char_bigram = character_bigrams_features(sentence1array, sentence2array)
    char_unigram = character_unigrams_features(sentence1array, sentence2array)
    word_unigram = word_unigram_features(sentence1array, sentence2array)
    word_bigram = word_bigram_features(sentence1array, sentence2array)

    # "CharacterBigramUnion", "CharacterBigramIntersection", "NumCharBigrams1", "NumCharBigrams2"
    features["CharacterBigramUnion"] = char_bigram["CharacterBigramUnion"]
    features["CharacterBigramIntersection"] = char_bigram["CharacterBigramIntersection"]
    features["NumCharBigrams1"] = char_bigram["NumCharBigrams1"]
    features["NumCharBigrams2"] = char_bigram["NumCharBigrams2"]

    # "CharacterUnigramUnion", "CharacterUnigramIntersection",
    #                                      "NumCharUnigrams1", "NumCharUnigrams2"
    features["CharacterUnigramUnion"] = char_unigram["CharacterUnigramUnion"]
    features["CharacterUnigramIntersection"] = char_unigram["CharacterUnigramIntersection"]
    features["NumCharUnigrams1"] = char_unigram["NumCharUnigrams1"]
    features["NumCharUnigrams2"] = char_unigram["NumCharUnigrams2"]

    # "SentenceUnigramUnion", "SentenceUnigramIntersection",
    # "NumSentUnigrams1", "NumSentUnigrams2"
    features["SentenceUnigramUnion"] = word_unigram["SentenceUnigramUnion"]
    features["SentenceUnigramIntersection"] = word_unigram["SentenceUnigramIntersection"]
    features["NumSentUnigrams1"] = word_unigram["NumSentUnigrams1"]
    features["NumSentUnigrams2"] = word_unigram["NumSentUnigrams2"]
    # "SentenceBigramUnion", "SentenceBigramIntersection",
    #                                      "NumSentBigrams1", "NumSentBigrams2"
    features["SentenceBigramUnion"] = word_bigram["SentenceBigramUnion"]
    features["SentenceBigramIntersection"] = word_bigram["SentenceBigramIntersection"]
    features["NumSentBigrams1"] = word_bigram["NumSentBigrams1"]
    features["NumSentBigrams2"] = word_bigram["NumSentBigrams2"]

    features["Meteor"] = meteor_scores(sentence1array, sentence2array)
    features["CosineSimilarity"] = vectorize_features(sentence1array, sentence2array)

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
penalty = ['l1', 'l2', 'elasticnet', 'none'],
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
valuesToDivide = list(range(1, 20))
C = [i / 10 for i in valuesToDivide]
# all_params = list(product(n_components, penalty, solver, C))
# parameters = [
#     {'pca__n_components': [n_components],
#      'logistic_Reg__solver': [solver],
#      'logistic_Reg__penalty': [penalty],
#      'logistic_Reg__C': [C]
#      } for n_components, penalty, solver, C in all_params
#     if not (penalty == 'l1' and solver in ["sag", "lbfgs", "newton-cg"])
#     and not (penalty == 'none' and solver in ["liblinear"])
#     and not (penalty == 'elasticnet' and solver in ["sag", "lbfgs", "newton-cg", "liblinear"])
# ]
Grid_params = [
    {'logistic_Reg__solver': ['saga'],
     'logistic_Reg__penalty': ['elasticnet', 'l1', 'l2', 'none'],
     'pca__n_components': n_components,
     'logistic_Reg__C': C},
    {'logistic_Reg__solver': ['newton-cg'],
     'logistic_Reg__penalty': ['l2', 'none'],
     'pca__n_components': n_components,
     'logistic_Reg__C': C},
    {'logistic_Reg__solver': ['lbfgs'],
     'logistic_Reg__penalty': ['l2', 'none'],
     'pca__n_components': n_components,
     'logistic_Reg__C': C},
    {'logistic_Reg__solver': ['liblinear'],
     'logistic_Reg__penalty': ['l1', 'l2'],
     'pca__n_components': n_components,
     'logistic_Reg__C': C},
    {'logistic_Reg__solver': ['sag'],
     'logistic_Reg__penalty': ['l2', 'none'],
     'pca__n_components': n_components,
     'logistic_Reg__C': C}
]

clf_GS = GridSearchCV(pipe, Grid_params, verbose=2)
clf_GS.fit(X, y)

print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(clf_GS.best_estimator_.get_params()['logistic_Reg'])
print("Optimized logistic regression model accuracy:" + str(clf_GS.score(Xdev, ydev)))

# Best Penalty: none
# Best C: 0.1
# Best Number Of Components: 8
# LogisticRegression(C=0.1, penalty='none', solver='newton-cg')
# Optimized logistic regression model accuracy:0.6443381180223285

# Code to find the optimal Support Vector Classifier Model
# pipeSVC = sklearn.pipeline.Pipeline(steps=[('std_slc', StandardScaler()),
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
# SVC optimal model finder is missing linear model found naturally for some reason, but
# Best kernel: linear
# Best C: 0.5623413251903491
# Best Number Of Components: 11
# SVC(C=0.5623413251903491, kernel='linear')
# Best SVC model accuracy: 0.6555023923444976
# Linear SVC Accuracy: 0.6586921850079744

#
# linearSVC = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.5))
# linearSVC.fit(X, y)
# print("Linear SVC Accuracy: " + str(linearSVC.score(Xdev, ydev)))
