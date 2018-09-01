import os
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib3

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


def import_wikitext(window_size=301, lines=100):
    
    WINDOW_SIZE = window_size
    LINES = lines
    
    if (WINDOW_SIZE%2 != 1):
        raise ValueError('WINDOW_SIZE must be odd!')
    
    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    url_add = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt'
    text = http.request('GET', url_add)
    
    text = text.data.decode('utf-8').split()
    
    text_len = len(text)
    first_start = 0
    last_start = text_len - WINDOW_SIZE
    print('text length:', text_len)
    
    seq_list = []
    pred_list = []
    for start in range(first_start, last_start):
        seq = ''
        for w in text[start:start+WINDOW_SIZE//2]:
            seq += w + ' '
        for w in text[start+WINDOW_SIZE//2+1:start+WINDOW_SIZE]:
            seq += w + ' '
        seq = seq.strip()
        pred = text[(start+WINDOW_SIZE//2)]
        seq_list.append(seq)
        pred_list.append(pred)
        if LINES > 0 and start >= LINES-1:
            break
            
    df = pd.DataFrame({'tag': pred_list, 'statement': seq_list})
    
    return df