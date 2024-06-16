import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # fully connected layer with output = 1

    def forward(self, x):
        batch_size = x.size(0) # get batch size bc input size is 1

        # initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        # get output
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :])

        return out