import csv
import string

from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords

DELIMITER = ','

MODEL_FILE = 'doc2vec_model.txt'
BLACK_LIST = ['.', ',', '?', ')', '(']

stemmer = SnowballStemmer(language="english")  # stemmer
punc = str.maketrans(string.punctuation, ' '  * len(string.punctuation))


def tokenize(text):
    #tokens = word_tokenize(text.lower())

    tokens = text.lower().split()

    # remove blacklisted tokens and stopwords
    #tokens = [token for token in tokens if not token in BLACK_LIST]  # - set(stopwords.words('english')))

    #if stem:
    #    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


class TaggedQuestions(object):

    def __init__(self, train_filename, stem=False):
        self.trainFile = train_filename
        #self.testFile = test_filename
        #self.stem = stem

        '''
        self.questions = []
        for question in self:
            self.questions.append(question)

        self.words = []
        for question in self.questions:
            self.words = self.words + question.words
        '''

    # TODO implement it with QuoraQuestionPair
    def __iter__(self):
        # training set
        with open(self.trainFile, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=DELIMITER)

            for row in reader:
                q1 = row['question1']
                q2 = row['question2']

                yield TaggedDocument(words=word_tokenize(q1), tags=[row['qid1']])
                yield TaggedDocument(words=word_tokenize(q2), tags=[row['qid2']])

        '''
        if self.testFile != "":
            # testing set
            with open(self.testFile, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=DELIMITER)

                for row in reader:
                    q1 = row['question1']
                    q2 = row['question2']

                    yield TaggedDocument(words=word_tokenize(q1), tags=['TEST_%s_Q1' % row['test_id']])
                    yield TaggedDocument(words=word_tokenize(q2), tags=['TEST_%s_Q2' % row['test_id']])
        '''

    '''
    def sentences(self):
        return self.questions

    def words(self):
        return self.words
    '''


class QuoraQuestionPairs(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        # training set
        with open(self.filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            for row in reader:
                yield row

    @staticmethod
    def training_set(filename):
        return QuoraQuestionPairs(filename)

    @staticmethod
    def testing_set(filename):
        return QuoraQuestionPairs(filename)


class QuoraQuestions(object):

    def __init__(self, quora_q_pairs):
        self.rows = quora_q_pairs

    def __iter__(self):
       for row in self.rows:
           yield row['question1']
           yield row['question2']

    @staticmethod
    def testing_set(filename):
        return QuoraQuestions(QuoraQuestionPairs.testing_set(filename))

    @staticmethod
    def training_set(filename):
        return QuoraQuestions(QuoraQuestionPairs.training_set(filename))



