import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.insert(0, parent_dir)

from utilities import accuracy


class LSTMClassification(nn.Module):
    def __init__(self, device, batch_size, input_size=1, hidden_size=4, num_stacked_layers=1, bidirectional=False, output_logits=True):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.output_logits = output_logits
        self.device = device
        self.num_directions = 2 if bidirectional else 1

        # self.hidden = self.init_hidden(batch_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, bidirectional=bidirectional) # already includes activation layers
        self.fc = nn.Linear(hidden_size*self.num_directions, 1) # fully connected layer with output = 1


    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_stacked_layers*self.num_directions, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_stacked_layers*self.num_directions, batch_size, self.hidden_size).to(self.device))


    def forward(self, x):
        '''
        Args:
            x: torch.Tensor with shape (batch_size, seq_len, input_size) (if batch_first=True, else (seq_len, batch_size, input_size))

        Returns:
            out: torch.Tensor with shape (batch_size, 1)
        '''

        # print(f'input shape: {x.shape}')

        # in case the last batch is smaller than the batch size, reset the hidden state
        # or in case the data in production is not batched
        # batch_size = x.size(0)
        # if batch_size != self.hidden[0].size(1):
        #     print('Batch size did not match')
        #     self.hidden = self.init_hidden(batch_size)

        batch_size = x.size(0) # get batch size bc input size is 1

        h0 = torch.zeros(self.num_stacked_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)

        # Forward pass through LSTM
        out, (h0, c0) = self.lstm(x, (h0, c0))

        # Save the new hidden and cell state for the next forward pass
        # self.hidden = (h0.clone().detach(), c0.clone().detach())

        # run output through fully connected layer
        # out[:, -1, :] is essentially the last hidden state of the last layer of the LSTM
        logits = self.fc(out[:, -1, :]) 

        if self.output_logits:
            return logits
        
        pred_probs = torch.sigmoid(logits) # apply sigmoid activation function to get probabilities
        return pred_probs
