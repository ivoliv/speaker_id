import numpy as np
import pandas as pd

def in_ipynb():
    try:
        cfg = get_ipython().config 
        return True
    except NameError:
        return False

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

def calc_miss_next_wds(y, y_preds):
    miss_next_word = 0
    for ex in range(y.size(1)):
        next_w_idx = np.argmax(y[:,ex] == 1).item() - 1
        if next_w_idx == -1:
            next_w_idx = y.size(0)-1
        next_w = y[next_w_idx,ex].item()
        next_w_pred = y_preds[next_w_idx,ex].item()
        miss_next_word += next_w != next_w_pred
    
    return miss_next_word