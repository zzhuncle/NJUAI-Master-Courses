import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.hidden_len = (self.seq_len + 2 * self.pred_len) // 2
        
        if self.individual:
            self.Linear = nn.ModuleList()
            self.Linear1 = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.hidden_len))
                self.Linear1.append(nn.Linear(self.hidden_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.hidden_len)
            self.Linear1 = nn.Linear(self.hidden_len, self.pred_len)

    def forward(self, x, embed=False):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            embedding = torch.zeros([x.size(0),self.hidden_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                embedding[:,:,i] = self.Linear[i](x[:,:,i])
                output[:,:,i] = self.Linear1[i](embedding[:,:,i])
            x = output
        else:
            embedding = self.Linear(x.permute(0,2,1))
            x = self.Linear1(embedding).permute(0,2,1)
        x = x + seq_last
        if embed:
            return x, embedding
        return x # [Batch, Output length, Channel]