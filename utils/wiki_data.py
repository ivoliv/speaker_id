from utils.data_import import Vocab
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class ITokList():

    def __init__(self, ilist):

        self.itoklist = ilist
        self.batch_matrix = None  # Need to call batchify
        self.batch_start_end = None  # Need to call batchify

    def __len__(self):

        return len(self.itoklist)
    
    def replace_token(self, token, vocab):
        
        # This method should only be called from the corpus class,
        # in particular the corpus.remove_token() method
        
        unk_i = vocab.stoi['<unk>']
        tok_i = vocab.stoi[token]
        self.itoklist = [unk_i if x == tok_i else x for x in self.itoklist]
    
    def replace_itoken_list(self, i_rem_map):
        
        # This method should only be called from the corpus class,
        # in particular the corpus.remove_token() method
        
        n_sub = 0
        for i, itok in enumerate(self.itoklist):
            self.itoklist[i] = i_rem_map[itok]
            if itok != i_rem_map[itok]:
                n_sub += 1
       
        return n_sub

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

            end = min(nrows-1, start + seq_len)

            self.batch_start_end.append((start, end))

        return

    def batch_stats(self):

        df = pd.DataFrame(self.batch_start_end)
        df['len'] = df[1] - df[0]
        print(df['len'].describe())

        return df['len']

class WikiCorpus():
    def __init__(self, file_path='./wikitext-2', lines=100, vocab_file=''):
        self.vocab = Vocab()
        self.imported_vocab = False
        
        self.specials = ['<pad>', '<eol>', '<unk>', '<upcase>']

        if vocab_file != '':
            self.vocab.import_vocab(vocab_file)
            print('Imported vocab:  {:,}'.format(len(self.vocab)))
            self.imported_vocab = True

        tok, l = self.tokenize(file_path, 'wiki.train.tokens', lines)
        self.train = ITokList(tok)
        print('Generated train: {:,} tokens ({:,} lines)'.format(len(self.train), l), flush=True)

        tok, l = self.tokenize(file_path, 'wiki.valid.tokens', lines)
        self.valid = ITokList(tok)
        print('Generated valid: {:,} tokens ({:,} lines)'.format(len(self.valid), l), flush=True)

        tok, l = self.tokenize(file_path, 'wiki.test.tokens', lines)
        self.test = ITokList(tok)
        print('Generated test:  {:,} tokens ({:,} lines)'.format(len(self.test), l), flush=True)

        if self.imported_vocab == False:
            print('Generated vocab: {:,}'.format(len(self.vocab)))

        print('Generated oov:   {:.1%}'.format(self.vocab.freq['<unk>'] /
                                               (len(self.train) + len(self.valid) + len(self.test))), flush=True)

    def tokenize(self, file_path, filename, lines):

        with open(os.path.join(file_path, filename), encoding='utf8') as f:

            for s in self.specials:
                if s not in self.vocab.stoi.keys(): 
                    self.vocab.add_token(s)
                    
            doc = f.readlines()

            text = ''

            for l, line in enumerate(doc):
                text += (line.replace('=', '').replace('.', ' . ').replace(',', ' , ')\
                    .replace("'", " '").replace('-', ' - ').replace('?', ' ? ')\
                    .replace('[', ' [ ').replace(']', ' ] ').replace('!', ' ! ')\
                    .replace('(', ' ( ').replace(')', ' ) ').replace('\"', ' \" ')\
                 ).strip() + ' <eol> '
                if lines > 0 and l >= lines:
                    break

            ilist = []

            for char in text.split():
                lchar = char.lower()
                if char != lchar:
                    ilist.append(self.vocab.add_token('<upcase>'))
                ilist.append(self.vocab.add_token(lchar))

        return ilist, l

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
    
    def remove_token(self, token):
        
        if token in self.specials:
            print('Error: not allowed to remove special token', token)
            return
        
        self.train.replace_token(token, self.vocab)
        self.valid.replace_token(token, self.vocab)
        self.test.replace_token(token, self.vocab)
        
        self.vocab.remove_token(token)
        
    def remove_token_list(self, token_list):
        
        for token in token_list:
            if token in self.specials:
                print('Note: keeping special token', token)
                token_list.remove(token)
                
        print('Processing {:,} tokens for removal.'.format(len(token_list)))
        
        print('Updating corpus vocabulary...', end=' ', flush=True)
        i_rem_map, n_rem = self.vocab.remove_token_list(token_list)
        
        print('{:,} removed.\nReplacing tokens in train...'.format(n_rem), end=' ', flush=True)
        n_rem = self.train.replace_itoken_list(i_rem_map)
        
        print('{:,} substitutions.\nReplacing tokens in valid...'.format(n_rem), end=' ', flush=True)
        n_rem = self.valid.replace_itoken_list(i_rem_map)
        
        print('{:,} substitutions.\nReplacing tokens in test...'.format(n_rem), end=' ', flush=True)
        n_rem = self.test.replace_itoken_list(i_rem_map)
        
        print('{:,} substitutions.\nFinal vocabulary size: {:,} tokens'.format(n_rem, len(self.vocab)), flush=True)


class WikiTextDataset(Dataset):

    def __init__(self, itoklist):
        self.itoklist = itoklist

    def __len__(self):
        return len(self.itoklist.batch_start_end)

    def __getitem__(self, idx):
        # pdb.set_trace()

        start = self.itoklist.batch_start_end[idx][0]
        end = self.itoklist.batch_start_end[idx][1]

        x = self.itoklist.batch_matrix[start:end, :]
        y = self.itoklist.batch_matrix[start + 1:end + 1, :]

        return x, y.contiguous()
