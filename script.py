
# coding: utf-8

# In[1]:


import utils.data_import as data_import
import utils.ml_utils as ml_utils

import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext import data, vocab


import os, sys
import pdb


# In[2]:


emb_dim = 100
hidden_dim = 100
num_linear = 3

batch_size = 128

device = 'cpu'


# In[3]:


data_path = '/Users/ivoliv/AI/insera/data/SpeechLabelingService'
ORIG_DATA = False  # True: original data, False: IMDB dataset


# In[4]:


if ORIG_DATA:
    train_file = 'training.txt'
    train, valid = data_import.normalize_and_split(data_path, train_file, test_size=.20)
else:
    train_file = '/Users/ivoliv/Datasets/imbd_csv/movie_reviews.csv'
    train, valid = data_import.import_imbd(train_file, to=10000, test_size=.20)


# In[5]:


data_import.create_split_files(data_path, train, valid)


# In[6]:


data_path = '/Users/ivoliv/AI/insera/data/SpeechLabelingService/data'


# In[7]:


TEXT = data.Field(sequential=True, tokenize='spacy', lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)


# In[8]:


datafields = [('tag', None),
              ('statement', TEXT),
              ('tag_id', LABEL)]

trn, vld = data.TabularDataset.splits(
    path=data_path,
    train='train.csv', validation='valid.csv',
    format='csv',
    skip_header=True,
    fields=datafields)


# In[9]:


TEXT.build_vocab(trn, vld, vectors='glove.6B.'+str(emb_dim)+'d')
LABEL.build_vocab(trn, vld)


# In[11]:


n_classes = len(dict(LABEL.vocab.freqs).keys())
print('Number of classes:', n_classes)


# In[ ]:


print('len(trn):', len(trn))
print('len(vld):', len(vld))
print(vars(trn[0]))
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)


# In[12]:


print(TEXT.vocab.vectors.shape)
vocab_size = len(TEXT.vocab)
TEXT.vocab.vectors[TEXT.vocab.stoi['the']]


# In[13]:


train_iter, val_iter = data.BucketIterator.splits(
    datasets=(trn, vld),
    batch_sizes=(batch_size, batch_size),
    sort_key=lambda x: len(x.statement),
    sort_within_batch=False,
    repeat=False
)


# In[14]:


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
            yield (X, y)


# In[15]:


train_dl = BatchGenerator(train_iter, 'statement', 'tag_id')
valid_dl = BatchGenerator(val_iter, 'statement', 'tag_id')


# In[16]:


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
        


# In[17]:


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


# In[18]:


class simpleRNN(nn.Module):
    def __init__(self, pretrained_vec, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        self.embedding.weight.requires_grad = False
        
        self.rnn = nn.RNN(emb_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, seq):
        emb = self.embedding(seq)
        out, hid = self.rnn(emb)
        out = self.fc(hid.squeeze(0))
        
        return F.log_softmax(out, dim=-1)


# In[19]:


model = CNN(TEXT.vocab.vectors,
            emb_dim=emb_dim,
            n_channels_per_filter=100,
            filter_sizes=[3, 4, 5]
           ).to(device)


# In[20]:


loss_func = nn.NLLLoss()


# In[21]:


print(model)


# In[22]:


import tqdm
 
opt = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.NLLLoss()
 
epochs = 10

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


# In[23]:




# In[24]:



