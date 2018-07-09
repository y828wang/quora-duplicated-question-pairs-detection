import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.spatial.distance import cosine, jaccard, matching
from scipy.stats import pearsonr
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from train_tfidf_model import tokenize

TFIDF_MODEL_FILE = 'models/bow_tfidf.pkl'
BIN_MODEL_FILE = 'models/bow_bin.pkl'


TRAIN_FILE = 'input/train.csv' #"output/train_balanced.csv"
TEST_FILE = 'input/test.csv' #"../test.csv"

TRAIN_OUTPUT_FILE = "features/bow_train.csv"
TEST_OUTPUT_FILE = "features/bow_test.csv"


def real_jaccard(ta, tb):
    return np.dot(ta, tb) / (LA.norm(ta) + LA.norm(tb) - np.dot(ta, tb))


def sym_kl_div(ta, tb):
    # smooth vectors
    ta += 1.
    tb += 1.

    pi1 = ta / (ta + tb)
    pi2 = tb / (ta + tb)
    t = (pi1 * ta) + (pi2 * tb)
    kl = np.sum((pi1 * ta * np.log(ta / t)) + (pi2 * tb * np.log(tb / t)))

    return kl


def compute_tfidf_features(sparse1, sparse2):
    nparray1 = sparse1.toarray()[0]
    nparray2 = sparse2.toarray()[0]

    features = pd.Series({
        'bow_tfidf_sum1': np.sum(sparse1),
        'bow_tfidf_sum2': np.sum(sparse2),
        'bow_tfidf_mean1': np.mean(sparse1),
        'bow_tfidf_mean2': np.mean(sparse2),
        'bow_tfidf_cosine': cosine(nparray1, nparray2),
        'bow_tfidf_jaccard': real_jaccard(nparray1, nparray2),
        'bow_tfidf_sym_kl_divergence': sym_kl_div(nparray1, nparray2),
        'bow_tfidf_pearson': pearsonr(nparray1, nparray2)[0]
    })

    return features


def compute_bin_features(sparse1, sparse2):
    nparray1 = sparse1.toarray()[0]
    nparray2 = sparse2.toarray()[0]

    features = pd.Series({
        'bow_bin_cosine': cosine(nparray1, nparray2),
        'bow_bin_jaccard': jaccard(nparray1, nparray2),
        'bow_bin_hamming': matching(nparray1, nparray2)
    })

    return features


def extract_features(df, tfidf_model, bin_model):
    df['q1_tfidf_bow'] = list(tfidf_model.transform(df.question1.astype(str)))
    df['q2_tfidf_bow'] = list(tfidf_model.transform(df.question2.astype(str)))
    df['q1_bin_bow'] = list(bin_model.transform(df.question1.astype(str)))
    df['q2_bin_bow'] = list(bin_model.transform(df.question2.astype(str)))

    tfidf_features = df.apply(lambda r: compute_tfidf_features(r.q1_tfidf_bow, r.q2_tfidf_bow), axis=1)
    bin_features = df.apply(lambda r: compute_bin_features(r.q1_bin_bow, r.q2_bin_bow), axis=1)

    features = pd.concat([tfidf_features, bin_features], axis=1)

    return features.fillna(.0)


def main():
    print('loading tfidf models...')
    tfidf_model = joblib.load(TFIDF_MODEL_FILE)
    bin_model = joblib.load(BIN_MODEL_FILE)  # binary bag of word representation

    # assert tokenize('') == [] # check whether tokenize is imported

    if TRAIN_FILE != "":
        print('embedding training output...')

        df = pd.read_csv(TRAIN_FILE)
        features = extract_features(df, tfidf_model, bin_model)
        features.to_csv(TRAIN_OUTPUT_FILE, index=False)

        print('bow_tfidf_cosine AUC:', roc_auc_score(df['is_duplicate'], 1 - features['bow_tfidf_cosine']))
        print('bow_tfidf_jaccard AUC:', roc_auc_score(df['is_duplicate'], features['bow_tfidf_jaccard']))
        print('bow_tfidf_pearson AUC:', roc_auc_score(df['is_duplicate'], features['bow_tfidf_pearson']))
        print('bow_tfidf_sym_kl_divergence AUC:', roc_auc_score(df['is_duplicate'],
                                                                1 - features['bow_tfidf_sym_kl_divergence']))
        print('bow_bin_cosine AUC:', roc_auc_score(df['is_duplicate'], 1 - features['bow_bin_cosine']))
        print('bow_bin_jaccard AUC:', roc_auc_score(df['is_duplicate'], 1 - features['bow_bin_jaccard']))
        print('bow_bin_hamming AUC:', roc_auc_score(df['is_duplicate'], 1 - features['bow_bin_hamming']))

    if TEST_FILE != "":
        print('embedding testing output...')
        df = pd.read_csv(TEST_FILE)
        features = extract_features(df, tfidf_model, bin_model)
        features.to_csv(TEST_OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()