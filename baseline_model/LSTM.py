import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, device, input_size=1, hidden_size=4, num_stacked_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True) # already includes activation layers
        self.fc = nn.Linear(hidden_size, 1) # fully connected layer with output = 1

    def forward(self, x):
        # input has shape (batch_size, seq_len, input_size) (if batch_first=True, else (seq_len, batch_size, input_size)
        # seq_len is the number of time steps (lookback = 7 equals 7 time steps)
        # input_size is the number of features per time step (1 feature per time step equals input_size = 1)

        batch_size = x.size(0) # get batch size bc input size is 1

        # initial hidden state
        # -> remembers short term dependencies of the sequence
        # influences the current output
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        # initial cell state
        # -> remembers long term dependencies of the sequence
        # some of this information is passed to the next cell state which then influences the next output
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0)) # get output of LSTM layer
        out = self.fc(out[:, -1, :]) # run output through fully connected layer
        return out