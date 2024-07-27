import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # print(f'Hiden state shape: {h0.shape}')
        # print(f'Hidden state: {h0}')
        # print(f'Out passed shape: {out[:, -1, :].shape}')
        # print(f'Out passed: {out[:, -1, :]}')
        # print(f'Out shape: {out.shape}')
        # print(f'Out: {out}')

        # run output through fully connected layer
        # out[:, -1, :] is essentially the last hidden state of the last layer of the LSTM
        logits = self.fc(out[:, -1, :]) 

        if self.output_logits:
            return logits
        
        pred_probs = torch.sigmoid(logits) # apply sigmoid activation function to get probabilities
        return pred_probs
    

class CNNClassification(nn.Module):
    
    def __init__(self, output_logits=True, verbose=False):
        super().__init__()
        self.output_logits = output_logits
        self.verbose = verbose
        # N, 5, 12
        
        self.conv1 = nn.Conv1d(5, 5, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(5, 5, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv1d(5, 5, kernel_size=4, stride=1, padding=0)
        self.linear = nn.Linear(5, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        print(f'Input after unsqueeze and permute: {x.shape}') if self.verbose else None

        # Encoder
        x = F.relu(self.conv1(x))
        print(f'After conv1: {x.shape}') if self.verbose else None

        x = F.relu(self.conv2(x))
        print(f'After conv2: {x.shape}') if self.verbose else None

        x = F.relu(self.conv3(x))
        print(f'After conv3: {x.shape}') if self.verbose else None

        logits = self.linear(x)
        print(f'After linear: {logits.shape}') if self.verbose else None

        if self.output_logits:
            return logits
        
        pred_probs = torch.sigmoid(logits) # apply sigmoid activation function to get probabilities
        return pred_probs