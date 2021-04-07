import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Encoder(nn.Module):
    def __init__(self, num_rnn_layers=1, rnn_hidden_size=128, dropout=0):
        super(Encoder,self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=(3,4),stride=(3,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(4,3),stride=(4,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=(4,2),stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.gru = nn.GRU(128,rnn_hidden_size,num_rnn_layers,
                        batch_first=True,
                        dropout=dropout)
        self.linear = nn.Linear(128,rnn_hidden_size*num_rnn_layers)

        self.layer4 = nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)
        self.layer5 = nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)
        self.layer6 = nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)

    def forward(self,x,hidden):
        h0 = hidden
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).squeeze(2)
        out = out.transpose(1,2)
        out,hidden = self.gru(out,h0)
        return out,hidden

    def initHidden(self,batch_size,use_cuda=False):
        h0 = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size))
        if use_cuda:
            return (h0.cuda())
        else:
            return h0

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn,self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: decode hidden state, (batch_size , N)
        :param encoder_outputs: encoder's all states, (batch_size,T,N)
        :return: weithed_context :(batch_size,N), alpha:(batch_size,T)
        """
        hidden_expanded = hidden.unsqueeze(2) #(batch_size,N,1)

        energy = torch.bmm(encoder_outputs,hidden_expanded).squeeze(2)

        alpha = nn.functional.softmax(energy, dim=1)
        weighted_context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted_context,alpha

class RNNAttnDecoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size,
                 output_size, num_rnn_layers=1, dropout=0.):
        super(RNNAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(input_vocab_size + hidden_size, hidden_size,
                            num_rnn_layers, batch_first=True,
                            dropout=dropout)

        self.wc = nn.Linear(2 * hidden_size, hidden_size)#,bias=False)
        self.ws = nn.Linear(hidden_size,output_size)

        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(input_vocab_size, input_vocab_size)
        fix_embedding = torch.from_numpy(np.eye(input_vocab_size, input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad=False

    def forward(self, input, last_ht, last_hidden, encoder_outputs):
        '''
        :se
        :param input: (batch_size,)
        :param last_ht: (obatch_size,hidden_size)
        :param last_hidden: (batch_size,hidden_size)
        :param encoder_outputs: (batch_size,T,hidden_size)
        '''
        embed_input = self.embedding(input)
        rnn_input = torch.cat((embed_input,last_ht),1)
        output, hidden = self.gru(rnn_input.unsqueeze(1),last_hidden)
        output = output.squeeze(1)

        weighted_context, alpha = self.attn(output,encoder_outputs)
        ht = self.tanh(self.wc(torch.cat((output,weighted_context),1)))
        output = self.ws(ht)
        return output, ht, hidden, alpha
