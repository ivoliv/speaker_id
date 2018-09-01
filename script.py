
# coding: utf-8

# In[2]:


import utils.data_import as data_import
import utils.ml_utils as ml_utils

import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data, vocab

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os, sys
import pdb
import pandas as pd


# In[3]:


emb_dim = 200
hidden_dim = 300
num_linear = 3

batch_size = 16

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    print('Device:', torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print('No cuda.')


# In[4]:


org_data_path = '/Users/ivoliv/AI/insera/data/SpeechLabelingService'
ORIG_DATA = 2  # 0: original data, 1: IMDB dataset, 2: wikitext


# In[5]:


if ORIG_DATA == 0:
    train_file = 'training.txt'
    train, valid = data_import.normalize_and_split(org_data_path, train_file, test_size=.20)
elif ORIG_DATA == 1:
    train_file = '/Users/ivoliv/Datasets/imbd_csv/movie_reviews.csv'
    train, valid = data_import.import_imbd(train_file, to=10000, test_size=.20)
elif ORIG_DATA == 2:
    df = data_import.import_wikitext(window_size=201, lines=500)
    train, valid = data_import.create_splits(df, test_size=.20)


# In[6]:


valid.head()


# In[7]:


data_import.create_split_files(org_data_path, train, valid)


# In[8]:


data_path = '/Users/ivoliv/AI/insera/data/SpeechLabelingService/data'


# In[9]:


TEXT = data.Field(sequential=True, tokenize='spacy', lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)


# In[10]:


datafields = [('tag', None),
              ('statement', TEXT),
              ('tag_id', LABEL)]

train, test = data.TabularDataset.splits(
    path=data_path,
    train='train.csv', validation='valid.csv',
    format='csv',
    skip_header=True,
    fields=datafields)


# In[11]:


TEXT.build_vocab(train, test, vectors='glove.6B.'+str(emb_dim)+'d')
LABEL.build_vocab(train, test)


# In[12]:


n_classes = len(dict(LABEL.vocab.freqs).keys())
print('Number of classes:', n_classes)


# In[13]:


print('len(trn):', len(train))
print('len(test):', len(test))
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
#print(vars(train[0]))
#print(LABEL.vocab.stoi)


# In[14]:


print(TEXT.vocab.vectors.shape)
vocab_size = len(TEXT.vocab)
TEXT.vocab.vectors[TEXT.vocab.stoi['the']]


# In[15]:


trn, vld = train.split(0.7)
print('len(trn):', len(trn))
print('len(vld):', len(vld))
print('len(test):', len(test))


# In[16]:


train_iter, val_iter, test_iter = data.BucketIterator.splits(
    datasets=(trn, vld, test),
    batch_sizes=(batch_size, batch_size, batch_size),
    sort_key=lambda x: len(x.statement),
    sort_within_batch=False,
    repeat=False
)


# In[17]:


class BatchGenerator:
    def __init__(self, dl, x_field, y_fields):
        self.dl, self.x_field, self.y_fields = dl, x_field, y_fields
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_fields)
            #y = torch.cat([getattr(batch, feat).unsqueeze(1) 
            #               for feat in self.y_fields], dim=1).float()
            if cuda:
                X = X.cuda()
            yield (X, y)


# In[18]:


train_dl = BatchGenerator(train_iter, 'statement', 'tag_id')
valid_dl = BatchGenerator(val_iter, 'statement', 'tag_id')


# In[19]:


# Requires padding to be set to 100
class SimpleForward(nn.Module):
    def __init__(self, pretrained_vec, emb_dim=50):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        self.embedding.weight.requires_grad = False
        
        self.linear_layer = nn.Linear(emb_dim*100, 128)
        self.linear_out = nn.Linear(128, n_classes)
        
    def forward(self, seq):
        
        seq = seq.permute(1, 0)
        emb = self.embedding(seq)
        
        long_layer = emb.view(-1, emb_dim*100)
        
        out = F.relu(self.linear_layer(long_layer))
        out = self.linear_out(out)
        
        return F.log_softmax(out, dim=-1)
        


# In[20]:


class CNN(nn.Module):
    def __init__(self, pretrained_vec, emb_dim, n_channels_per_filter, filter_sizes):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        self.embedding.weight.requires_grad = False
        
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_channels_per_filter,
                                kernel_size=(filter_sizes[0], emb_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_channels_per_filter,
                                kernel_size=(filter_sizes[1], emb_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_channels_per_filter,
                                kernel_size=(filter_sizes[2], emb_dim))
        
        self.fc = nn.Linear(len(filter_sizes)*n_channels_per_filter, n_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, seq):
        
        seq = seq.permute(1, 0)
        emb = self.embedding(seq)
        
        #print(seq.shape)
        #print(emb.shape)
        
        emb = emb.unsqueeze(1) # introduce 'channel 1'
        
        #print('emb:', emb.shape)
        
        out0 = F.relu(self.conv_0(emb).squeeze(3))
        out1 = F.relu(self.conv_1(emb).squeeze(3))
        out2 = F.relu(self.conv_2(emb).squeeze(3))
        
        #print('out0:', out0.shape)
        
        pooled_0 = F.max_pool1d(out0, out0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(out1, out1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(out2, out2.shape[2]).squeeze(2)
        
        #print('pooled0:', pooled_0.shape)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        
        #print('cat:', cat.shape)
        
        out = self.fc(cat)
        
        return F.log_softmax(out, dim=-1)


# In[21]:


class simpleRNN(nn.Module):
    def __init__(self, pretrained_vec, emb_dim, hidden_dim, change_emb=True):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        self.embedding.weight.requires_grad = change_emb
        
        self.rnn = nn.RNN(emb_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, seq):
        # seq dims: [seq len, batch size]
        emb = self.embedding(seq)
        # emb dims: [seq len, batch size, emb dim]
        out, hid = self.rnn(emb)
        # out dims: [seq len, batch size, hidden_dim]
        # hid dims: [1, batch size, hidden_dim]
        # hid.squeeze(0) -> [batch size, hidden_dim]
        out = self.fc(hid.squeeze(0))
        # out dims: [batch size, n_classes]
        sm = F.log_softmax(out, dim=-1)
        # sm dims: [batch size, n_classes]
        
        return sm


# In[22]:


class simpleLSTM(nn.Module):
    def __init__(self, pretrained_vec, emb_dim, hidden_dim, n_layers=1,
                 change_emb=True, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        self.embedding.weight.requires_grad = change_emb
        
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, dropout=dropout)
        
        self.fc = nn.Linear(2*hidden_dim, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq):
        #pdb.set_trace()
        # seq dims: [seq len, batch size]
        
        emb = self.embedding(seq)
        # emb dims: [seq len, batch size, emb dim]
        
        out, (hid, cel) = self.lstm(emb)
        
        # out dims: [seq len, batch size, hidden_dim]
        # hid dims: [2*n_layers, batch size, hidden_dim]
        # cel dims: [2*n_layers, batch size, hidden_dim]
        # out[-1,:,:hd] -> [batch size, hidden_dim]  (last time step hidden vector)
        # out[0,:,hd:] <- [batch size, hidden_dim]  (first time step hidden vector)
        conc = torch.cat((out[-1,:,:hidden_dim], out[0,:,hidden_dim:]), dim=1)
        
        conc = self.dropout(conc)
        
        out = self.fc(conc)
        # out dims: [batch size, n_classes]
        sm = F.log_softmax(out, dim=-1)
        # sm dims: [batch size, n_classes]
        
        return sm


# In[23]:


#model = CNN(TEXT.vocab.vectors,
#            emb_dim=emb_dim,
#            n_channels_per_filter=100,
#            filter_sizes=[3, 4, 5]
#           )


# In[24]:


model = simpleLSTM(TEXT.vocab.vectors,
                   emb_dim=emb_dim,
                   hidden_dim=hidden_dim,
                   n_layers=num_linear,
                   change_emb=True,
                   dropout=0
                  )
if cuda:
    model = model.cuda()


# In[25]:


#x, y = next(iter(train_dl))
#pred = model(x)


# In[26]:


len(vars(trn[0])['statement'])


# In[27]:


loss_func = nn.NLLLoss()


# In[28]:


print(model)


# In[ ]:


import tqdm
 
opt = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.NLLLoss()
 
epochs = 100

missclass = []
losses = []
 
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # turn on training mode
    
    num_vals = 0
    num_correct = 0
    
    #pdb.set_trace()
    
    #for x, y in train_dl: 
    for x, y in tqdm.tqdm(train_dl): 
        opt.zero_grad()
 
        preds = model(x)
        loss = loss_func(preds, y.long())

        loss.backward()
        opt.step()
 
        running_loss += loss.item() * x.size(0)
    
        _, y_pred = torch.max(preds, dim=1)
        num_correct += torch.sum(y == y_pred).item()
        num_vals += len(y.float())
        
    #pdb.set_trace()
        
    missclass_tr = 1 - num_correct / num_vals
 
    epoch_loss = running_loss / len(trn)
    
    num_vals = 0
    num_correct = 0

    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval() # turn on evaluation mode
    with torch.no_grad():
        for x, y in tqdm.tqdm(valid_dl):
            preds = model(x)
            loss = loss_func(preds, y.long())
            val_loss += loss.item() * x.size(0)

            _, y_pred = torch.max(preds, dim=1)
            num_correct += torch.sum(y == y_pred).item()
            num_vals += len(y.float())
        
    #pdb.set_trace()
        
    missclass_te = 1 - num_correct / num_vals
    val_loss /= len(vld)
    
    missclass.append((missclass_tr, missclass_te))
    losses.append((epoch_loss, val_loss))
    
    print('Epoch: {}, Loss: [{:.4f}, {:.4f}], Miss: [{:.2%}, {:.2%}]'          .format(epoch, epoch_loss, val_loss, missclass_tr, missclass_te))
    sys.stdout.flush()


# In[ ]:


plt.plot(losses)
plt.legend(['train', 'valid'])


# In[ ]:


plt.plot(missclass)
plt.legend(['train', 'valid'])


# In[ ]:


import spacy
nlp = spacy.load('en')

def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed)
    if cuda:
        tensor = tensor.cuda()
    tensor = tensor.unsqueeze(1)
    prediction = model(tensor)
    return prediction.item()


# In[ ]:


predict_sentiment("This film is terrible")

