from utils.data_import import Vocab
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import utils as skutils
import pdb

class ITokListFrame():

    def __init__(self, ilist_df, text_col, tag_col):

        self.itoklist_df = ilist_df
        self.text_col = text_col
        self.tag_col = tag_col
        self.itoklist_df.loc[:,'len'] = self.itoklist_df[self.text_col].apply(lambda r: len(r))

        self.batch_matrix = None  # Need to call batchify
        self.batch_start_end = None  # Need to call batchify

    def __len__(self):

        return self.itoklist_df.shape[0]

    def show_itoklist(self, to=5):

        i = 0
        for idx, itok in self.itoklist_df.iterrows():
            i += 1
            print('##', self.tag_col,':', itok[0])
            print('##', self.text_col,':', itok[1])
            print()
            if i >= to:
                break
        return

    def show_stoklist(self, vocab, to=5):

        i = 0
        for idx, itok in self.itoklist_df.iterrows():
            i += 1
            print('##', self.tag_col, ':', itok[0])
            print('##', self.text_col, ':', itok[1])
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
        list_of_texts = []

        for idx, row in batch_df.iterrows():

            row_array = row[text][:seq_len]
            for padding in range(len(row_array), seq_len):
                row_array.append(vocab.stoi['<pad>'])
            row_array.append(int(row[tag]))

            list_of_texts.append(row_array)

        self.batch_matrix = torch.tensor(list_of_texts).view(batch_size, -1).t_()
        
        end = 0
        self.batch_start_end = []
        for start in range(0, self.batch_matrix.size()[0], seq_len+1):
            end = start + seq_len+1
            self.batch_start_end.append([start, end])

        if not suppress_print:
            print(' Number of batches: {:,}'.format(nbatches))
            print(' Preserved texts: {:,}'.format(batch_df.shape[0]))
            print(' Matrix size:      ', self.batch_matrix.size())

        return

    def batchify_2(self, pad_idx, batch_size, seq_length=70,
                   sort=True, rand_range=5, suppress_print=False):

        seq_len = seq_length

        nbatches = len(self.itoklist_df) // batch_size

        if sort:
            self.itoklist_df = self.itoklist_df.sort_values(by='len', ascending=True)

        self.batch_start_end = []
        end = 0

        for batch_no in range(0, nbatches):

            batch_seq_len = min(seq_len, self.itoklist_df.iloc[(1 + batch_no) * batch_size - 1]['len']) \
                            + np.random.randint(-rand_range, rand_range+1)
            batch_seq_len = max(5, batch_seq_len)

            start = end
            end = start + batch_seq_len + 1

            batch_texts = []

            for i in range(batch_no * batch_size, (1 + batch_no) * batch_size):

                row_array = self.itoklist_df.iloc[i][self.text_col]
                row_array = row_array[:batch_seq_len]
                for padding in range(len(row_array), batch_seq_len):
                    row_array.append(pad_idx)
                row_array.append(int(self.itoklist_df.iloc[i][self.tag_col]))

                batch_texts.append(row_array)

            self.batch_start_end.append([start, end])

            this_batch_matrix = torch.tensor(batch_texts).view(batch_size, -1).t_()
            if batch_no == 0:
                self.batch_matrix = this_batch_matrix
            else:
                self.batch_matrix = torch.cat((self.batch_matrix, this_batch_matrix), dim=0)

        if not suppress_print:
            print(' Number of batches: {:,}'.format(nbatches))
            print(' Preserved texts: {:,}'.format(nbatches * batch_size))
            print(' Matrix size:      ', self.batch_matrix.size())

        return

    def batch_stats(self):

        df = pd.DataFrame(self.batch_start_end)
        df['len'] = df[1] - df[0]
        print(df['len'].describe())

        return df['len']

class ImdbCorpus():
    def __init__(self, filename, lines=100, vocab_file='', text_col='review', tag_col='sentiment'):
        self.vocab = Vocab()
        self.imported_vocab = False

        self.classes = Vocab()

        if vocab_file != '':
            self.vocab.import_vocab(vocab_file)
            print('Imported vocab:  {:,}'.format(len(self.vocab)))
            self.imported_vocab = True

        tok, l = self.tokenize(filename, lines, text_col, tag_col)
        print('Read total of: {:,} lines from imdb file.'.format(l), flush=True)

        self.n_classes = len(self.classes)
        print('Number of classes:', self.n_classes, end=': ')

        print(self.classes.stoi)

        train, valid = train_test_split(tok, train_size=.7, test_size=.3)
        self.train = ITokListFrame(train, text_col, tag_col)
        print('Generated train: {:,} lines'.format(len(self.train)), flush=True)

        valid, test = train_test_split(valid, train_size=.5, test_size=.5)
        self.valid = ITokListFrame(valid, text_col, tag_col)
        print('Generated valid: {:,} lines'.format(len(self.valid)), flush=True)

        self.test = ITokListFrame(test, text_col, tag_col)
        print('Generated test:  {:,} lines'.format(len(self.test)), flush=True)

        if self.imported_vocab == False:
            print('Generated vocab: {:,}'.format(len(self.vocab)))

    def tokenize(self, filename, lines, text_col, tag_col):

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
            text = (r[1][text_col].replace('/><br', ' ')\
                    .replace('/>', ' ').replace('.', ' .') + ' <eol> ').strip()
            sent = r[1][tag_col].strip()

            ilist = []
            for char in text.split():
                lchar = char.lower()
                if char != lchar:
                    ilist.append(self.vocab.add_token('<upcase>'))
                ilist.append(self.vocab.add_token(lchar))

            tuplelist.append((self.classes.add_token(sent),
                              ilist))

        return pd.DataFrame(tuplelist, columns=[tag_col, text_col]), len(tuplelist)

    def batchify(self, pad_idx, batch_size, seq_length=70, rand_range=5,
                 sort=True, suppress_print=False):

        if not suppress_print:
            print('Batch size:       ', batch_size)
            print('Sequence length:  ', seq_length)
            print('Batchifying train...')
        self.train.batchify_2(pad_idx=pad_idx, batch_size=batch_size, seq_length=seq_length,
                              sort=sort, rand_range=rand_range, suppress_print=suppress_print)
        if not suppress_print:
            print('Batchifying valid...')
        self.valid.batchify_2(pad_idx=pad_idx, vocab=self.vocab, batch_size=batch_size, seq_length=seq_length,
                              sort=sort, rand_range=rand_range, suppress_print=suppress_print)
        if not suppress_print:
            print('Batchifying test... ')
        self.test.batchify_2(pad_idx=pad_idx, vocab=self.vocab, batch_size=batch_size, seq_length=seq_length,
                             sort=sort, rand_range=rand_range, suppress_print=suppress_print)

        return


class ImdbTextDataset(Dataset):

    def __init__(self, itoklistFrame, pad_idx, batch_size,
                 seq_length, rand_range, sort=True, rebatch_and_shuffle=False):

        self.itoklistFrame = itoklistFrame
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rand_range = rand_range
        self.sort = sort
        self.rebatch_and_shuffle = rebatch_and_shuffle

        self.itoklistFrame.batchify_2(pad_idx=pad_idx, batch_size=batch_size, seq_length=seq_length,
                                      rand_range=rand_range, sort=sort, suppress_print=False)
        self.itoklistFrame.batch_start_end = skutils.shuffle(self.itoklistFrame.batch_start_end)

    def __len__(self):
        return len(self.itoklistFrame.batch_start_end)

    def __getitem__(self, idx):
        # pdb.set_trace()

        if idx == len(self) and self.rebatch_and_shuffle:
            self.itoklistFrame.batchify_2(pad_idx=self.pad_idx, batch_size=self.batch_size,
                                          seq_length=self.seq_length,
                                          rand_range=self.rand_range, sort=self.sort,
                                          suppress_print=True)
            self.itoklistFrame.batch_start_end = skutils.shuffle(self.itoklistFrame.batch_start_end)

        start = self.itoklistFrame.batch_start_end[idx][0]
        end = self.itoklistFrame.batch_start_end[idx][1]

        x = self.itoklistFrame.batch_matrix[start:end-1, :]
        y = self.itoklistFrame.batch_matrix[end-1, :]

        return x, y


