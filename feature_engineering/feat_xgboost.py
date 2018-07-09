import datetime
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'input/train.csv'
TRAIN_FEATURES = "features/train.csv"
TEST_FEATURES = "features/test.csv"

TRAIN_PREDICTION = 'output/bst_train_pred.csv'
SUBMISSION_FILE = 'output/bst_test_pred.csv'

POS_PROP = 0.1746


def train_test_split_rebalance(features):
    train_features, valid_features = train_test_split(features, test_size=0.1, random_state=4242)

    # rebalance validation set
    pos_samples = valid_features[valid_features['is_duplicate'] == 1]
    neg_samples = valid_features[valid_features['is_duplicate'] == 0]

    pos_count = int(len(neg_samples) / (1 - POS_PROP) - len(neg_samples))
    valid_features = pd.concat([neg_samples, pos_samples[:pos_count]], ignore_index=True)

    # rebalance training set
    pos_samples = pd.concat([train_features[train_features['is_duplicate'] == 1], pos_samples[pos_count:]], ignore_index=True)
    neg_samples = train_features[train_features['is_duplicate'] == 0]

    needed = int(len(pos_samples) / POS_PROP - len(pos_samples))
    multiplier = needed // len(neg_samples)
    remainder = needed % len(neg_samples)

    train_features = pd.concat([neg_samples] * multiplier + [neg_samples[:remainder], pos_samples], ignore_index=True)

    return [train_features, valid_features]


def predict(features, bst, filename):
    p_test = bst.predict(features)

    print('writing predictions...')
    predictions = pd.DataFrame()
    predictions['test_id'] = range(0, p_test.shape[0])
    predictions['is_duplicate'] = p_test
    predictions = predictions.fillna(POS_PROP)
    predictions.to_csv(filename, index=False)


def main():

    print('loading training set...')
    train_data = pd.read_csv(TRAIN_DATA)
    train_features = pd.read_csv(TRAIN_FEATURES)

    print('features: {0}'.format(list(train_features)))

    train_features = train_features.fillna(.0)
    train_features['is_duplicate'] = train_data.is_duplicate

    print('rebalancing and creating cross-validation set...')
    train_t_features, train_cv_features = train_test_split_rebalance(train_features)

    print('label mean in training set is {0}'.format(train_t_features.is_duplicate.mean()))
    print('label mean in cross-validation set is {0}'.format(train_cv_features.is_duplicate.mean()))

    d_train = xgb.DMatrix(train_t_features.drop(['is_duplicate'], axis=1),
                          label=train_t_features.is_duplicate)
    d_valid = xgb.DMatrix(train_cv_features.drop(['is_duplicate'], axis=1),
                          label=train_cv_features.is_duplicate)

    params = {'objective': 'binary:logistic',
              'eval_metric': ['logloss'],
              'eta': 0.02,
              'max_depth': 8,
              "subsample": 0.7,
              "min_child_weight": 1,
              "colsample_bytree": 0.5,
              "silent": 1,
              "seed": 1632,
              'tree_method': 'exact'
              }

    bst = xgb.train(params, d_train, 2000, [(d_train, 'train'), (d_valid, 'cross-validation')],
                    early_stopping_rounds=50, verbose_eval=10)

    # saving model
    print('saving bst model...')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    joblib.dump(bst, 'models/bst-{0}.model'.format(timestamp))

    # making predictions
    print('predicting training output...')
    d_train = xgb.DMatrix(train_features.drop(['is_duplicate'], axis=1))
    predict(d_train, bst, TRAIN_PREDICTION)

    print('loading testing output...')
    test_features = pd.read_csv(TEST_FEATURES)
    test_features = test_features.fillna(.0)
    d_test = xgb.DMatrix(test_features)

    print('predicting testing output...')
    predict(d_test, bst, SUBMISSION_FILE)

if __name__ == '__main__':
    main()





