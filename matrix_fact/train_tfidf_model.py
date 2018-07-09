import pandas as pd
from gensim.parsing import PorterStemmer
from nltk import word_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


TRAIN_FILE = 'input/train.csv'
TEST_FILE = 'input/test.csv'

MODEL_TFIDF_FILE = 'models/bow_tfidf.pkl'
MODEL_BIN_FILE = 'models/bow_bin.pkl'

'''
stemmer = PorterStemmer()


def tokenize(text):
    try:
        tokens = [stemmer.stem(token) for token in word_tokenize(text.lower())]
    except IndexError:
        print('warning, fail to stem {0}'.format(text))
        tokens = word_tokenize(text)
    return tokens
'''


def tokenize(txt):
    return str(txt).lower().split()


def build_vectorizer(binary):
    use_idf = not binary
    norm = None if binary else u'l2'
    vectorizer = TfidfVectorizer(lowercase=True,
                            max_df=0.95,
                            min_df=2,
                            ngram_range=(1, 1),
                            stop_words='english',
                            tokenizer=tokenize,
                            sublinear_tf=False,
                            use_idf=use_idf,
                            norm=norm,
                            binary=binary)
    return vectorizer


def main():

    tfidf_model = build_vectorizer(binary=False)
    bin_model = build_vectorizer(binary=True)

    if TRAIN_FILE != '':
        print('fitting training set...')

        train_data = pd.read_csv(TRAIN_FILE)
        train_data.question1 = train_data.question1.astype(str)
        train_data.question2 = train_data.question2.astype(str)

        tfidf_model.fit(train_data.question1)
        tfidf_model.fit(train_data.question2)
        bin_model.fit(train_data.question1)
        bin_model.fit(train_data.question2)

    if TEST_FILE != '':
        print('fitting testing set...')
        test_data = pd.read_csv(TEST_FILE)

        test_data.question1 = test_data.question1.astype(str)
        test_data.question2 = test_data.question2.astype(str)

        tfidf_model.fit(test_data.question1)
        tfidf_model.fit(test_data.question2)
        bin_model.fit(test_data.question1)
        bin_model.fit(test_data.question2)

    joblib.dump(tfidf_model, MODEL_TFIDF_FILE)
    joblib.dump(bin_model, MODEL_BIN_FILE)

if __name__ == '__main__':
    main()

