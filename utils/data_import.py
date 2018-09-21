import pickle
import pdb
import collections
from tqdm import tqdm

class Vocab():

    def __init__(self):

        self.stoi = {}
        self.itos = []
        self.freq = {}
        self.locked_vocab = False

    def __len__(self):
        return len(self.stoi)
    
    def remove_token(self, token):
        
        # This method should only be called from the corpus class,
        # in particular the corpus.remove_token() method
        
        if self.locked_vocab:
            print('Error: vocab locked')
            return
        
        itoken = self.stoi[token]
        
        self.freq['<unk>'] += self.freq[token]
            
        #for i in range(itoken, len(self)):
        #    self.stoi[self.itos[i]] -= 1

        #self.itos.remove(token)
        self.itos[itoken] = '<unk_del>'
        del self.stoi[token] 
        del self.freq[token]

    def remove_token_list(self, token_list):
        
        # This method should only be called from the corpus class,
        # in particular the corpus.remove_token() method
        
        if self.locked_vocab:
            print('Error: vocab locked')
            return
        
        new_stoi = {}
        new_itos = []
        ireplaced = {}
        nreplaced = 0
        
        for token in tqdm(self.stoi.keys()):
            if token in token_list:
                self.freq['<unk>'] += self.freq[token]
                del self.freq[token]
                ireplaced[self.stoi[token]] = self.stoi['<unk>']
                nreplaced += 1
            else:
                new_stoi[token] = len(new_stoi)
                new_itos.append(token)
                self.freq[token] = self.freq[token]
                ireplaced[self.stoi[token]] = new_stoi[token]
            
        self.stoi = new_stoi
        self.itos = new_itos
        
        return ireplaced, nreplaced
        
    def add_token(self, token):

        if token not in self.stoi.keys():
            if self.locked_vocab:
                token = '<unk>'
            else:
                self.stoi[token] = len(self.stoi)
                self.itos.append(token)
                self.freq[token] = 0

        self.freq[token] += 1

        return self.stoi[token]

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

    def export_vocab(self, vocab_file='vocab.p'):

        print('Exporting vocab to {}... '.format(vocab_file), end='', flush=True)
        fileObject = open(vocab_file, 'wb')
        pickle.dump(self, fileObject)
        fileObject.close()
        print('Done.')

        return

    def import_vocab(self, vocab_file='vocab.p'):

        print('Importing vocab from {}... '.format(vocab_file), end='', flush=True)

        fileObject = open(vocab_file, 'rb')
        data = pickle.load(fileObject)

        self.stoi = data.stoi
        self.itos = data.itos
        # imported vocab starts with zero fequency counts
        for tok in self.stoi:
            self.freq[tok] = 0
        self.locked_vocab = True

        read_error = False

        try:
            assert(self.stoi['<pad>'] >= 0)
        except:
            print("Error: vocab must contain '<pad>' token.")
            read_error = True

        try:
            assert(self.stoi['<eol>'] >= 0)
        except:
            print("Error: vocab must contain '<eol>' token.")
            read_error = True

        try:
            assert(self.stoi['<unk>'] >= 0)
        except:
            print("Error: vocab must contain '<unk>' token.")
            read_error = True

        try:
            assert(self.stoi['<upcase>'] >= 0)
        except:
            print("Error: vocab must contain '<upcase>' token.")
            read_error = True

        assert(read_error == False)

        print('Done.')

        fileObject.close()

        return


