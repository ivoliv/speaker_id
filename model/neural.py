import torch.nn as nn
import pdb

class lang_model_LSTM(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hidden_dim, n_layers=1,
                 dropout=0, bidirectional=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ndir = (2 if bidirectional == True else 1)

        self.embedding = nn.Embedding(vocab_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(self.ndir * hidden_dim, vocab_dim)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self, initrange=0.1):

        self.embedding.weight.data.uniform_(-initrange, initrange)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                param.data.zero_()
            elif 'weight' in name:
                param.data.uniform_(-initrange, initrange)

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, seq):

        # pdb.set_trace()
        # seq dims: [seq len, batch size]

        emb = self.embedding(seq)
        # emb dims: [seq len, batch size, emb dim]

        out, (hid, cel) = self.lstm(emb)

        # out dims: [seq len, batch size, hidden_dim]
        # hid dims: [2*n_layers, batch size, hidden_dim]
        # cel dims: [2*n_layers, batch size, hidden_dim]
        # out[-1,:,:hd] -> [batch size, hidden_dim]  (last time step hidden vector)
        # out[0,:,hd:] <- [batch size, hidden_dim]  (first time step hidden vector)
        # contatenation of last time period, whole batch, forward (first) chunck of hidden units
        #   and the first time period, whole batch, backward (last) chunck of hidden units
        #   (pytorch concatenates hidden units across dim #2 for bidirectional LSTM)
        # if self.ndir == 2:
        #    conc = torch.cat((out[-1,:,:self.hidden_dim], out[0,:,self.hidden_dim:]), dim=1)
        # else:
        #    conc = out[0,:,self.hidden_dim]

        output = self.dropout(out)

        output = self.fc(output)

        # sm dims: [batch size, n_classes]
        # sm = F.log_softmax(output, dim=-1)
        # return sm

        return output