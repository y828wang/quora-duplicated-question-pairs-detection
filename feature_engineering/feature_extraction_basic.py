import difflib
import string

import pandas as pd
from nltk import pos_tag, PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score

from utils.clean_data import clean_txt

TRAIN_FILE = 'input/train.csv'
TEST_FILE = 'input/test.csv' #''output/test_cleaned.csv'

TRAIN_OUTPUT_FILE = "features/basic_train.csv"
TEST_OUTPUT_FILE = "features/basic_test.csv"

seq = difflib.SequenceMatcher()
stops = stopwords.words('english')
stemmer = PorterStemmer()
punct_map = str.maketrans('', '', string.punctuation)


def seq_ratio(st1, st2):
    seq.set_seqs(st1.lower(), st2.lower())
    return seq.ratio()


def extract_noun(words):
    return [w for w, t in pos_tag(words) if t[:1] in ['N']]


def word_share(words1, words2):
    words1 = set(words1)
    words2 = set(words2)
    union_len = len(words1) + len(words2)
    return len(words1.intersection(words2)) / union_len if union_len > 0 else .0


def tokenize(txt):
    #tokens = word_tokenize(str(txt).lower())
    #tokens = [w for w in tokens if w not in stops]
    tokens = str(txt).lower().split()
    tokens = [w for w in tokens if w not in stops]
    return tokens


def extract_features(df):
    features = pd.DataFrame()

    # length features
    print('extracting length features...')

    df['q1_words'] = df.question1.map(tokenize)
    df['q2_words'] = df.question2.map(tokenize)

    features['q1_str_len'] = df.question1.map(lambda q: len(str(q)))
    features['q2_str_len'] = df.question2.map(lambda q: len(str(q)))
    features['abs_str_diff_len'] = features.q1_str_len.rsub(features.q2_str_len).abs()

    features['q1_word_len'] = df.q1_words.map(lambda ws: len(ws))
    features['q2_word_len'] = df.q2_words.map(lambda ws: len(ws))

    features['q1_no_space_str_len'] = df.question1.map(lambda q: len(str(q).replace(' ', '')))
    features['q2_no_space_str_len'] = df.question2.map(lambda q: len(str(q).replace(' ', '')))

    features['common_word_count'] = df.apply(lambda r: len(set(r.q1_words).intersection(set(r.q2_words))), axis=1)

    # extract noun
    print('extracting nouns...')

    df['q1_nouns'] = df.q1_words.map(extract_noun)
    df['q2_nouns'] = df.q2_words.map(extract_noun)

    # match ratios
    print('extracting match ratios...')

    features['word_match'] = df.apply(lambda r: word_share(r.q1_words, r.q2_words), axis=1)
    features['diff_seq_ratio'] = df.apply(lambda r: seq_ratio(clean_txt(r.question1), clean_txt(r.question2)), axis=1)
    #features['len_match'] = features.abs_str_diff_len / (features.q1_str_len + features.q2_str_len)
    features['noun_match'] = df.apply(lambda r: word_share(r.q1_nouns, r.q2_nouns), axis=1)

    return features.fillna(.0)


if __name__ == "__main__":

    if TRAIN_FILE != "":
        print('loading training output...')
        df = pd.read_csv(TRAIN_FILE)

        print('embedding training output...')
        features = extract_features(df)

        print('word_match accuracy:', roc_auc_score(df['is_duplicate'], features['word_match']))
        print('noun_match accuracy:', roc_auc_score(df['is_duplicate'], features['noun_match']))
        print('diff_seq_ratio accuracy:', roc_auc_score(df['is_duplicate'], features['diff_seq_ratio']))
        # print('len_match accuracy:', 1 - roc_auc_score(df['is_duplicate'], features['len_match']))

        features.to_csv(TRAIN_OUTPUT_FILE, index=False)

    if TEST_FILE != "":
        print('loading testing output...')
        df = pd.read_csv(TEST_FILE)

        print('embedding testing output...')
        features = extract_features(df)

        features.to_csv(TEST_OUTPUT_FILE, index=False)
