import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def get_splits(data_path):

    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'valid.csv'))

    return train, test


def create_counts(dataset):

    count_vect = CountVectorizer()
    print('Counting and feature vectorizing...', end=' ', flush=True)
    X_train_counts = count_vect.fit_transform(list(dataset['statement']))
    print('Done.', flush=True)

    return count_vect, X_train_counts


def transform_and_tfidf(counts_dataset):

    print('Transform and tfidf...', end=' ', flush=True)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(counts_dataset)
    print('Done.', flush=True)

    return tfidf_transformer, X_train_tfidf


def train_model(X_tfidf, tags):

    print('Training...', end=' ', flush=True)
    clf = MultinomialNB().fit(X_tfidf, tags)
    print('Done.', flush=True)

    return clf


def calc_miss(clf, count_vect, tfidf_transformer, docs_set):
    print('Calculating missclassifications...', end=' ', flush=True)

    docs_new = list(docs_set['statement'])

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    comp = np.vstack((docs_set['tag_id'], predicted))
    diff = np.sum(comp[0, :] != comp[1, :])
    tot = len(comp[0, :])
    miss = diff / tot

    print('Done', flush=True)

    return miss


if __name__ == '__main__':

    data_path = '/Users/ivoliv/AI/insera/data/SpeechLabelingService/data'
    train, test = get_splits(data_path)

    count_vect, X_train_counts = create_counts(train)
    tfidf_transformer, X_train_tfidf = transform_and_tfidf(X_train_counts)

    clf = train_model(X_train_tfidf, train['tag_id'])

    tr_miss = calc_miss(clf, count_vect, tfidf_transformer, train)
    te_miss = calc_miss(clf, count_vect, tfidf_transformer, test)

    print('[{:.2%}, {:.2%}]'.format(tr_miss, te_miss))
