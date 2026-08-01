import re
import pickle

import distance
import numpy as np
import nltk
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words("english"))

SAFE_DIV = 0.0001

# NOTE: this file now uses a TF-IDF vectorizer (variable name kept as `cv` for
# compatibility with app.py, but it is actually a TfidfVectorizer object).
cv = pickle.load(open('cv.pkl', 'rb'))


def preprocess(q):
    q = str(q).lower().strip()

    q = q.replace('%', ' percent ')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000', 'b')
    q = q.replace(',000,000', 'm')
    q = q.replace(',000', 'k')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        " 'cause": "because", "could've": "could have", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
        "he'd": "he would", "he'll": "he will", "how'd": "how did", "how's": "how is",
        "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
        "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
        "let's": "let us", "mightn't": "might not", "mustn't": "must not",
        "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
        "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have", "wasn't": "was not", "we'd": "we would",
        "we're": "we are", "we've": "we have", "weren't": "were not",
        "what's": "what is", "where's": "where is", "who's": "who is",
        "won't": "will not", "wouldn't": "would not", "you'd": "you would",
        "you'll": "you will", "you're": "you are", "you've": "you have"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    q = BeautifulSoup(q, "html.parser").get_text()

    pattern = re.compile(r'\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def test_common_words(q1, q2):
    w1 = set(map(lambda x: x.lower().strip(), q1.split()))
    w2 = set(map(lambda x: x.lower().strip(), q2.split()))
    return len(w1 & w2)


def test_total_words(q1, q2):
    w1 = set(map(lambda x: x.lower().strip(), q1.split()))
    w2 = set(map(lambda x: x.lower().strip(), q2.split()))
    return len(w1) + len(w2)


def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features


def test_fetch_token_features(q1, q2):
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    if len(strs) == 0:
        length_features[2] = 0
    else:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features


def query_point_creator(q1, q2):
    # Preprocess both questions the same way the training data was preprocessed
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query = []

    # Length features (q1_len, q2_len)
    input_query.append(len(q1))
    input_query.append(len(q2))

    # Number of words features (q1_num_words, q2_num_words)
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    # Common words, total words, word share
    common_count = test_common_words(q1, q2)
    total_count = test_total_words(q1, q2)
    input_query.append(total_count)
    input_query.append(common_count)
    input_query.append(round(common_count / (total_count + SAFE_DIV), 2))

    # Fuzzy features
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    # Token features
    input_query.extend(test_fetch_token_features(q1, q2))

    # Length features
    input_query.extend(test_fetch_length_features(q1, q2))

    # TF-IDF vectors for both questions
    q1_vec = cv.transform([q1])
    q2_vec = cv.transform([q2])

    # Cosine similarity between the two TF-IDF vectors (both are L2-normalized
    # by TfidfVectorizer, so dot product == cosine similarity)
    cosine_sim = float(q1_vec.multiply(q2_vec).sum())
    input_query.append(cosine_sim)

    # TF-IDF bag-of-words features
    input_query.extend(q1_vec.toarray()[0])
    input_query.extend(q2_vec.toarray()[0])

    return np.array(input_query).reshape(1, -1)
