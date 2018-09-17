import pickle
import pdb

class Vocab():

    def __init__(self):

        self.stoi = {}
        self.itos = []
        self.freq = {}
        self.locked_vocab = False

    def __len__(self):
        return len(self.itos)

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


