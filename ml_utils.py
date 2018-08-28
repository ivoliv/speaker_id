import numpy as np
import pandas as pd


def show_sample(clf, count_vect, tfidf_transformer, dataset, to=10):
    sample = list(dataset['statement'])[:to]

    X_counts = count_vect.transform(sample)
    X_tfidf = tfidf_transformer.transform(X_counts)

    predicted = clf.predict(X_tfidf)

    comp = np.vstack((dataset['tag_id'][:to], predicted))

    comp = dataset.merge(pd.DataFrame(comp.T), left_index=True, right_index=True)

    print('\nCorrect classifications:')
    print(comp[comp[0] == comp[1]])
    print('\nMissclassified:')
    print(comp[comp[0] != comp[1]])

    return comp