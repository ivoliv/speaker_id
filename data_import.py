import os
import pandas as pd
from sklearn.model_selection import train_test_split


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

    train.to_csv(os.path.join(data_path, 'data', 'train.csv'), index=False)
    valid.to_csv(os.path.join(data_path, 'data', 'valid.csv'), index=False)
    print('Train and validation files written to disk.')
    print('Sizes:', train.shape, valid.shape)


if __name__ == '__main__':

    data_path = '/Users/ivoliv/AI/insera/data/SpeechLabelingService'
    train_file = 'training.txt'

    train, valid = normalize_and_split(data_path, 10)
    create_split_files(data_path, train, valid)
