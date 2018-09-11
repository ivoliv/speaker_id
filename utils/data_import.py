import os
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib3
import numpy as np

def read_from_file(data_path, train_file):

    with open(os.path.join(data_path, train_file), encoding='utf8') as f:
        train_doc = f.readlines()

    train = []
    for l in train_doc:
        str_list = l.split()
        tag = str_list[0].strip()
        text = l.replace(tag, '').strip()
        train.append((tag, text))

    return train


def create_splits(df, test_size=0.10):

    classes = df['tag'].unique()
    c_map = {j: i for i, j in enumerate(classes)}
    df['tag_id'] = df['tag'].apply(lambda x: c_map[x])

    train, valid = train_test_split(df, test_size=test_size,
                                    random_state=123, shuffle=True)

    return train, valid


def normalize_and_split(data_path, train_file, test_size=0.10):

    input_data = read_from_file(data_path, train_file)
    df = pd.DataFrame(input_data, columns=['tag', 'statement'])
    train, valid = create_splits(df, test_size)

    return train, valid


def import_imbd(train_file, to=None, test_size=0.10):

    df = pd.read_csv(train_file)
    if to:
        df = df[:to]
    df['tag'] = df['sentiment']
    df['statement'] = df['review']
    df = df.drop(['review', 'sentiment'], axis=1)

    train, valid = create_splits(df, test_size=test_size)

    return train, valid


def create_split_files(data_path, train, valid):

    print('Writing files to', os.path.join(data_path, 'data'))
    train.to_csv(os.path.join(data_path, 'data', 'train.csv'), index=False)
    valid.to_csv(os.path.join(data_path, 'data', 'valid.csv'), index=False)
    print('Train and validation files written to disk.')
    print('Sizes:', train.shape, valid.shape)
    
    return 


def import_wikitext(window_size=70, lines=100, prob_cut=0.05, cut_factor=0.50):
    
    WINDOW_SIZE = window_size
    LINES = lines
    
    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    url_add = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt'
    text = http.request('GET', url_add)
    
    text = text.data.decode('utf-8').split()
    
    text_len = len(text)
    first_start = 0
    last_start = text_len - WINDOW_SIZE - 1
    print('Original text length:  {:,} token sequence.'.format(text_len))
    
    seq_list = []
    pred_list = []
    ret_text_len = 0
    start = 0
    lines_so_far = 0
    while True:
        if start > last_start:
            break
        if LINES > 0 and lines_so_far >= LINES:
            break
            
        seq = ''
        pred = ''
        
        factor = np.random.choice([1, cut_factor], p=[1-prob_cut, prob_cut])
        WINDOW_SIZE = int(window_size * factor)
        WINDOW_SIZE = max(5, int(np.random.normal(WINDOW_SIZE, 5)))
        
        for w in text[start:start+WINDOW_SIZE]:
            seq += w + ' '
        seq = seq.strip()
        for w in text[start+1:start+WINDOW_SIZE+1]:
            pred += w + ' '
        pred = pred.strip()
        seq_list.append(seq)
        pred_list.append(pred)
        ret_text_len += len(seq) + len(pred)
        lines_so_far += 1
        start += WINDOW_SIZE + 1
            
    df = pd.DataFrame({'tag': pred_list, 'statement': seq_list})
    
    print('Generated text length: {:,} tokens.'.format(ret_text_len))
    
    return df