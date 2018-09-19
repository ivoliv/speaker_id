from utils.data_import import Vocab
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb

class ITokListFrame():

    def __init__(self, ilist_df):

        self.itoklist_df = ilist_df
        self.batch_matrix = None  # Need to call batchify
        self.batch_start_end = None  # Need to call batchify

    def __len__(self):

        return self.itoklist_df.shape[0]

    def show_itoklist(self, to=5):

        i = 0
        for idx, itok in self.itoklist_df.iterrows():
            i += 1
            print('## sentiment:', itok[0])
            print('## text:     ', itok[1])
            print()
            if i >= to:
                break
        return

    def show_stoklist(self, vocab, to=5):

        i = 0
        for idx, itok in self.itoklist_df.iterrows():
            i += 1
            print('## sentiment:', itok[0])
            print('## text:     ', end=' ')
            for w in itok[1]:
                print(vocab.itos[w], end=' ')
            print('\n', flush=True)
            if i >= to:
                break
        return

    def batchify(self, vocab, batch_size, seq_length=70, suppress_print=False):

        seq_len = seq_length

        nbatches = len(self.itoklist_df) // batch_size
        
        batch_df = self.itoklist_df[:nbatches * batch_size]
        list_of_reviews = []

        for idx, row in batch_df.iterrows():

            row_array = row['review'][:seq_len]
            for padding in range(len(row_array), seq_len):
                row_array.append(vocab.stoi['<pad>'])
            row_array.append(int(row['sentiment']))

            list_of_reviews.append(row_array)

        self.batch_matrix = torch.tensor(list_of_reviews).view(batch_size, -1).t_()
        
        end = 0
        self.batch_start_end = []
        for start in range(0, self.batch_matrix.size()[0], seq_len+1):
            end = start + seq_len+1
            self.batch_start_end.append([start, end])

        if not suppress_print:
            print(' Number of batches: {:,}'.format(nbatches))
            print(' Preserved reviews: {:,}'.format(batch_df.shape[0]))
            print(' Matrix size:      ', self.batch_matrix.size())

        return

    def batch_stats(self):

        df = pd.DataFrame(self.batch_start_end)
        df['len'] = df[1] - df[0]
        print(df['len'].describe())

        return df['len']

class ImdbCorpus():
    def __init__(self, filename='./movie_reviews.csv', lines=100, vocab_file=''):
        self.vocab = Vocab()
        self.imported_vocab = False

        self.classes = Vocab()

        if vocab_file != '':
            self.vocab.import_vocab(vocab_file)
            print('Imported vocab:  {:,}'.format(len(self.vocab)))
            self.imported_vocab = True

        tok, l = self.tokenize(filename, lines)
        print('Read total of: {:,} lines from imdb file.'.format(l), flush=True)

        self.n_classes = len(self.classes)
        print('Number of classes:', self.n_classes, end=': ')

        print(self.classes.stoi)

        train, valid = train_test_split(tok, train_size=.7, test_size=.3)
        self.train = ITokListFrame(train)
        print('Generated train: {:,} lines'.format(len(self.train)), flush=True)

        valid, test = train_test_split(valid, train_size=.5, test_size=.5)
        self.valid = ITokListFrame(valid)
        print('Generated valid: {:,} lines'.format(len(self.valid)), flush=True)

        self.test = ITokListFrame(test)
        print('Generated test:  {:,} lines'.format(len(self.test)), flush=True)

        if self.imported_vocab == False:
            print('Generated vocab: {:,}'.format(len(self.vocab)))

    def tokenize(self, filename, lines):

        if lines == 0:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename, nrows=lines)

        self.vocab.add_token('<pad>')
        self.vocab.add_token('<eol>')
        self.vocab.add_token('<unk>')
        self.vocab.add_token('<upcase>')

        tuplelist = []

        for r in df.iterrows():
            text = (r[1][0].replace('/><br', ' ')\
                    .replace('/>', ' ').replace('.', ' .') + ' <eol> ').strip()
            sent = r[1][1].strip()

            ilist = []
            for char in text.split():
                lchar = char.lower()
                if char != lchar:
                    ilist.append(self.vocab.add_token('<upcase>'))
                ilist.append(self.vocab.add_token(lchar))

            tuplelist.append((self.classes.add_token(sent),
                              ilist))

        return pd.DataFrame(tuplelist, columns=['sentiment', 'review']), len(tuplelist)

    def batchify(self, batch_size, seq_length=70, suppress_print=False):

        if not suppress_print:
            print('Batch size:       ', batch_size)
            print('Sequence length:  ', seq_length)
            print('Batchifying train...')
        self.train.batchify(self.vocab, batch_size, seq_length, suppress_print)
        if not suppress_print:
            print('Batchifying valid...')
        self.valid.batchify(self.vocab, batch_size, seq_length, suppress_print)
        if not suppress_print:
            print('Batchifying test... ')
        self.test.batchify(self.vocab, batch_size, seq_length, suppress_print)

        return


class ImdbTextDataset(Dataset):

    def __init__(self, itoklistFrame):
        self.itoklistFrame = itoklistFrame

    def __len__(self):
        return len(self.itoklistFrame.batch_start_end)

    def __getitem__(self, idx):
        # pdb.set_trace()

        start = self.itoklistFrame.batch_start_end[idx][0]
        end = self.itoklistFrame.batch_start_end[idx][1]

        x = self.itoklistFrame.batch_matrix[start:end-1, :]
        y = self.itoklistFrame.batch_matrix[end-1, :]

        return x, y
