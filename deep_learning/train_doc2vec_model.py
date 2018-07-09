from random import shuffle

from gensim.models.doc2vec import Doc2Vec

from utils.quora_question_pairs_helpers import TaggedQuestions

VEC_DIM = 400

TRAIN_FILE = 'train_sample.csv'
NUM_CORES = 2
ITER = 20

MODEL_FILE = 'models/doc2vec_model_test.txt'


def main():
    '''
    if len(sys.argv) != 4:
        print(USAGE)
        exit(0)
    '''

    train_file = TRAIN_FILE#sys.argv[1]
    #test_file = sys.argv[2]
    n_workers = NUM_CORES

    print('reading training output...')
    questions = [question for question in TaggedQuestions(train_file, stem=False)]

    # for review in reviews:
    #    print(review)

    #model = Doc2Vec(min_count=3, window=3, size=VEC_DIM, sample=1e-4, negative=15, workers=n_workers, dm=1)
    model = Doc2Vec(min_count=5, window=10, size=VEC_DIM, sample=1e-4, negative=5, workers=n_workers, dm=1)


    print('building vocabulary...')
    model.build_vocab(questions, keep_raw_vocab=True)

    for epoch in range(ITER):
        print("training doc2vec, iter {0}".format(epoch))
        shuffle(questions)
        model.train(questions)


    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save(MODEL_FILE)


if __name__ == "__main__":
    main()