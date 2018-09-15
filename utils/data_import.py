import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pdb

class ITokList():

    def __init__(self, ilist):

        self.itoklist = ilist
        self.batch_matrix = None  # Need to call batchify
        self.batch_start_end = None  # Need to call batchify

    def __len__(self):

        return len(self.itoklist)

    def show_itoklist(self, start=0, to=0):

        if to == 0:
            to = start + 200  # Default print 200 items

        for i, itok in enumerate(self.itoklist[start:to]):
            print(itok, end=' ')
            if i >= to:
                break

    def show_stoklist(self, vocab, start=0, to=0):

        if to == 0:
            to = start + 200  # Default print 200 items

        for i, itok in enumerate(self.itoklist[start:to]):
            print(vocab.itos[itok], end=' ')
            if i >= to:
                break

    def batchify(self, batch_size, seq_length=70, prob_cut=0.05, cut_factor=0.50):

        nrows = len(self.itoklist) // batch_size
        self.batch_matrix = torch.tensor(self.itoklist[:nrows*batch_size]).view(batch_size, nrows).t_()

        end = 0
        self.batch_start_end = []

        while end + 1 < nrows:

            start = end

            factor = np.random.choice([1, cut_factor], p=[1-prob_cut, prob_cut])
            seq_len = int(seq_length * factor)
            seq_len = max(5, int(np.random.normal(seq_len, 5)))

            end = min(nrows, start + seq_len)

            self.batch_start_end.append((start, end))

        return

    def batch_stats(self):

        df = pd.DataFrame(self.batch_start_end)
        df['len'] = df[1] - df[0]
        print(df['len'].describe())

        return df['len']

class Vocab():

    def __init__(self):

        self.stoi = {}
        self.itos = []
        self.freq = {}

    def __len__(self):
        return len(self.itos)

    def add_token(self, token):

        if token not in self.stoi.keys():
            self.stoi[token] = len(self.stoi)
            self.itos.append(token)
            self.freq[token] = 1
        else:
            self.freq[token] += 1

        return self.stoi[token]

    def most_frequent(self):

        return sorted(self.freq.items(), key=lambda kv: kv[1], reverse=True)

    def most_frequent(self, to=20):

        sort_list = sorted(self.freq.items(), key=lambda kv: kv[1], reverse=True)

        if to == 0 or to > len(sort_list):
            to = len(sort_list)

        return sort_list[:to]

    def least_frequent(self, to=20):

        sort_list = sorted(self.freq.items(), key=lambda kv: kv[1], reverse=False)

        if to == 0 or to > len(sort_list):
            to = len(sort_list)

        return sort_list[:to]


class Corpus():
    def __init__(self, file_path='./wikitext-2', lines=100):
        self.vocab = Vocab()
        self.train = ITokList(self.tokenize(file_path, 'wiki.train.tokens', lines))
        print('Generated train: {:,} tokens'.format(len(self.train)))
        self.valid = ITokList(self.tokenize(file_path, 'wiki.valid.tokens', lines))
        print('Generated valid: {:,} tokens'.format(len(self.valid)))
        self.test = ITokList(self.tokenize(file_path, 'wiki.test.tokens', lines))
        print('Generated test:  {:,} tokens'.format(len(self.test)))
        print('Generated vocab: {:,}'.format(len(self.vocab)))
        print('Generated oov:   {:.1%}'.format(self.vocab.freq['<unk>']/
                                               (len(self.train)+len(self.valid)+len(self.test))))

    def tokenize(self, file_path, filename, lines):

        with open(os.path.join(file_path, filename), encoding='utf8') as f:

            doc = f.readlines()

            text = ''

            for line in doc:
                text += line.replace('=', '').strip() + ' <eol> '

            ilist = []

            self.vocab.add_token('<pad>')
            self.vocab.add_token('<eol>')
            self.vocab.add_token('<unk>')
            self.vocab.add_token('<upcase>')

            for char in text.split():
                lchar = char.lower()
                if char != lchar:
                    ilist.append(self.vocab.add_token('<upcase>'))
                ilist.append(self.vocab.add_token(lchar))

        return ilist

    def batchify(self, batch_size, seq_length=70, prob_cut=0.05, cut_factor=0.50):

        print('Batchifying train...', end=' ', flush=True)
        self.train.batchify(batch_size, seq_length, prob_cut, cut_factor)
        print('Done. ', self.train.batch_matrix.shape, flush=True)
        print('Batchifying valid...', end=' ', flush=True)
        self.valid.batchify(batch_size, seq_length, prob_cut, cut_factor)
        print('Done. ', self.valid.batch_matrix.shape, flush=True)
        print('Batchifying test... ', end=' ', flush=True)
        self.test.batchify(batch_size, seq_length, prob_cut, cut_factor)
        print('Done. ', self.test.batch_matrix.shape, flush=True)

        return


class WikiTextDataset(Dataset):

    def __init__(self, itoklist):

        self.itoklist = itoklist

    def __len__(self):
        return len(self.itoklist.batch_start_end)

    def __getitem__(self, idx):
        #pdb.set_trace()

        start = self.itoklist.batch_start_end[idx][0]
        end = self.itoklist.batch_start_end[idx][1]

        x = self.itoklist.batch_matrix[start:end, :]
        y = self.itoklist.batch_matrix[start+1:end+1, :]

        return x, y.contiguous()
