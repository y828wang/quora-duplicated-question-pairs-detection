import string

import distance
import pandas as pd
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score

TRAIN_FILE = 'input/sample.csv'
TEST_FILE = ''#'input/test.csv' #'input/test.csv' #''output/test_cleaned.csv'

TRAIN_OUTPUT_FILE = "features/str_dist_train.csv"
TEST_OUTPUT_FILE = "features/str_dist_test.csv"

stops = stopwords.words('english')
stemmer = PorterStemmer()
punct_map = str.maketrans('', '', ',.?;:')


def smooth(tokens):
    if len(tokens) == 0:
        tokens = ['xyz']
    return tokens


def space_split(txt):
    tokens = str(txt).lower().split()
    tokens = [w for w in tokens if w not in stops]
    return smooth(tokens)


def stem(txt):
    '''
    remove stopwords, remove punctuation, stem words
    '''

    tokens = word_tokenize(str(txt).lower())
    tokens = [w for w in tokens if w not in stops and w not in string.punctuation]

    try:
        tokens = [stemmer.stem(w) for w in tokens]
    except IndexError:
        print('warn {0} fail to stem'.format(tokens))

    #print(tokens)

    return smooth(tokens)


def extract_features(df):
    features = pd.DataFrame()

    print('extracting space splitted sequence features...')

    df['q1_words'] = df.question1.map(space_split)
    df['q2_words'] = df.question2.map(space_split)

    features['str_leven1'] = df.apply(lambda r: distance.nlevenshtein(r.q1_words, r.q2_words, method=1), axis=1)
    features['str_leven2'] = df.apply(lambda r: distance.nlevenshtein(r.q1_words, r.q2_words, method=2), axis=1)
    features['str_jaccard'] = df.apply(lambda r: distance.jaccard(r.q1_words, r.q2_words), axis=1)
    #features['str_hamming'] = df.apply(lambda r: distance.hamming(r.q1_words, r.q2_words, normalized=True), axis=1)
    #features['str_sorensen'] = df.apply(lambda r: distance.jaccard(r.question1, r.question2), axis=1)

    print('extracting stemmed word sequence features...')

    df['q1_stems'] = df.question1.map(stem)
    df['q2_stems'] = df.question2.map(stem)

    features['stem_leven1'] = df.apply(lambda r: distance.nlevenshtein(r.q1_stems, r.q2_stems, method=1), axis=1)
    features['stem_leven2'] = df.apply(lambda r: distance.nlevenshtein(r.q1_stems, r.q2_stems, method=2), axis=1)
    features['stem_jaccard'] = df.apply(lambda r: distance.jaccard(r.q1_stems, r.q2_stems), axis=1)

    return features.fillna(.0)


if __name__ == "__main__":

    if TRAIN_FILE != "":
        print('loading training output...')
        df = pd.read_csv(TRAIN_FILE)

        print('embedding training output...')
        features = extract_features(df)

        print('str_leven1 accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['str_leven1']))
        print('str_leven2 accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['str_leven2']))
        print('str_jaccard accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['str_jaccard']))
        #print('str_hamming accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['str_hamming']))
        #print('str_sorensen accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['str_sorensen']))

        print('stem_leven1 accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['stem_leven1']))
        print('stem_leven2 accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['stem_leven2']))
        print('stem_jaccard accuracy:', roc_auc_score(df['is_duplicate'], 1 - features['stem_jaccard']))

        features.to_csv(TRAIN_OUTPUT_FILE, index=False)

    if TEST_FILE != "":
        print('loading testing output...')
        df = pd.read_csv(TEST_FILE)

        print('embedding testing output...')
        features = extract_features(df)

        features.to_csv(TEST_OUTPUT_FILE, index=False)
