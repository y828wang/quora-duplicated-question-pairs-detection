import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'input/train.csv'
TEST_DATA = 'input/test.csv'

TRAIN_PRE = 'output/train_cleaned.csv'
TEST_PRE = 'output/test_cleaned.csv'

punct_map = str.maketrans('', '', string.punctuation)


def clean_txt(text):
    text = str(text).lower()

    re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", ' ', text)  # removing non ASCII chars

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


if __name__ == '__main__':
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)

    # cleaning text
    print('cleaning text...')
    train_data['question1'] = train_data['question1'].fillna('not applied').map(clean_txt)
    train_data['question2'] = train_data['question2'].fillna('not applied').map(clean_txt)
    test_data['question1'] = test_data['question1'].fillna('not applied').map(clean_txt)
    test_data['question2'] = test_data['question2'].fillna('not applied').map(clean_txt)

    # write preprocessed output
    print('writing...')
    #train_data.to_csv(TRAIN_PRE, index=False)
    test_data.to_csv(TEST_PRE, index=False)
