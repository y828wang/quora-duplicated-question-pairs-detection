import pandas
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.metrics import roc_auc_score

TRAIN_FILE = "input/train.csv"
TEST_FILE =  "input/test.csv"

TRAIN_OUTPUT_FILE = "features/fuzz_train.csv"
TEST_OUTPUT_FILE = "features/fuzz_test.csv"

def extract_features(df):
    features = pd.DataFrame()
    features['fuzz_qratio'] = df.apply(lambda r: fuzz.QRatio(str(r.question1), str(r.question2)), axis=1)
    features['fuzz_wratio'] = df.apply(lambda r: fuzz.WRatio(str(r.question1), str(r.question2)), axis=1)
    features['fuzz_partial_ratio'] = df.apply(lambda r: fuzz.partial_ratio(str(r.question1), str(r.question2)), axis=1)
    features['fuzz_partial_token_set_ratio'] = df.apply(lambda r: fuzz.partial_token_set_ratio(str(r.question1), str(r.question2)), axis=1)
    features['fuzz_partial_token_sort_ratio'] = df.apply(lambda r: fuzz.partial_token_sort_ratio(str(r.question1), str(r.question2)), axis=1)
    return features


def main():
    if TRAIN_FILE != "":
        print('loading training output...')
        dataframe = pd.read_csv(TRAIN_FILE)
        print('{0} training output loaded'.format(len(dataframe)))

        print('embedding training output...')
        features = extract_features(dataframe)

        print('fuzz_qratio AUC:', roc_auc_score(dataframe['is_duplicate'], features['fuzz_qratio']))
        print('fuzz_wratio AUC:', roc_auc_score(dataframe['is_duplicate'], features['fuzz_wratio']))
        print('fuzz_partial_ratio AUC:', roc_auc_score(dataframe['is_duplicate'], features['fuzz_partial_ratio']))
        print('fuzz_partial_token_set_ratio AUC:', roc_auc_score(dataframe['is_duplicate'], features['fuzz_partial_token_set_ratio']))
        print('fuzz_partial_token_sort_ratio AUC:', roc_auc_score(dataframe['is_duplicate'], features['fuzz_partial_token_sort_ratio']))

        features.to_csv(TRAIN_OUTPUT_FILE, index=False)

    if TEST_FILE != "":
        print('loading testing output...')
        dataframe = pd.read_csv(TEST_FILE)
        print('{0} testing output loaded'.format(len(dataframe)))

        print('embedding testing output...')
        features = extract_features(dataframe)

        features.to_csv(TEST_OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()
