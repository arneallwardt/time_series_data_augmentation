import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

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
    


### TRAIN AND TEST LOOP ###

def train_one_epoch(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        device, 
        log_interval=100, 
        scheduler=None):
    
    '''Trains the model for one epoch and returns the average training loss. If a scheduler is provided, the learning rate is updated.'''
    
    model.train()
    running_train_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)  

        train_pred = model(x_batch)
        train_loss = criterion(train_pred, y_batch)
        running_train_loss += train_loss.item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if batch_index % log_interval == 0:
            
            # log training loss 
            avg_train_loss_across_batches = running_train_loss / log_interval
            # print(f'Batch: {batch_index}, Loss: {avg_train_loss_across_batches}')

            # update learning rate
            if(scheduler is not None):
                current_learning_rate = scheduler.get_last_lr()
                scheduler.step(avg_train_loss_across_batches)
                if current_learning_rate != scheduler.get_last_lr():
                    print(f'INFO: Scheduler updated Learning rate from ${current_learning_rate} to {scheduler.get_last_lr()}')

            running_train_loss = 0.0 # reset running loss


def validate_one_epoch(
        model, 
        test_loader, 
        criterion, 
        device):
    
    '''Validates the model and returns the average validation loss.'''
    
    model.eval()
    running_test_loss = 0.0

    with torch.inference_mode():
        for _, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            test_pred = model(x_batch)
            test_loss = criterion(test_pred, y_batch)
            running_test_loss += test_loss.item()

    # log validation loss
    avg_test_loss_across_batches = running_test_loss / len(test_loader)
    print(f'Validation Loss: {avg_test_loss_across_batches}')
    return avg_test_loss_across_batches


def train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        device,
        patience=10, 
        num_epochs=1000):
    
    '''Trains the model and returns the best validation loss aswell as the trained model. Stops training if the validation loss does not improve for patience epochs.'''
    
    best_validation_loss = np.inf
    num_epoch_without_improvement = 0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}')
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        current_validation_loss = validate_one_epoch(model, test_loader, criterion, device)
        
        # early stopping
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            num_epoch_without_improvement = 0
        else:
            print(f'INFO: Validation loss did not improve in epoch {epoch + 1}')
            num_epoch_without_improvement += 1

        if num_epoch_without_improvement >= patience:
            print(f'Early stopping after {epoch + 1} epochs')
            break

        print(f'*' * 50)

    return best_validation_loss, model



### DATA PREPROCESSING ###


def scale_data(np_array):
    '''Scales each feature individually using MinMaxScaler and returns the scaled numpy array aswell as the scaler used for scaling the closing price.'''
    n_features_per_timestep = np_array.shape[-1]
    scalers = []

    # scale each feature individually and save the scalers to inverse scale the data later
    for i in range(n_features_per_timestep):
        scalers.append(MinMaxScaler(feature_range=(0, 1))) 
        np_array[:, :, i] = scalers[i].fit_transform(np_array[:, :, i])

    return np_array, scalers[0]


def inverse_scale_data(np_array, scaler, seq_len):
    '''Inverse scales the data using the given scaler and returns the inverse scaled numpy array.'''
    # create dummies to match the required shape of the scaler and set the first column to the array to scale
    dummies = np.zeros((np_array.shape[0], seq_len))
    dummies[:, 0] = np_array.flatten()

    # inverse scale the data
    dummies_scaled = scaler.inverse_transform(dummies)

    # get only first column of the dummies_scaled array, since this is where the original data was
    np_array = dc(dummies_scaled[:, 0])

    print(f'Shape of the inverse scaled numpy array: {np_array.shape}')
    return np_array


def scale_data_same_scaler(np_array, scaler):
    '''CURRENTLY NOT IN USE: Scales features together using the given scaler and returns the scaled numpy array.'''
    n_samples = np_array.shape[0]  
    n_timesteps = np_array.shape[1]

    np_array = np_array.reshape(-1, 2)

    np_array = scaler.fit_transform(np_array)
    
    np_array = np_array.reshape(n_samples, n_timesteps, 2)

    return np_array


def train_test_split_to_tensor(np_array, split_ratio=0.95):
    '''Splits the data into train and test set, flips the column order of the features and converts them to tensors.'''

    X = np_array[:, :-1]
    X = torch.tensor(X, dtype=torch.float32)
    y = np_array[:, -1, 0] # only take the closing price as target, ignore the other features
    y = torch.tensor(y, dtype=torch.float32)

    split_index = int(len(X) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index].unsqueeze(1)
    y_test = y[split_index:].unsqueeze(1)

    print(f'Shape of X_train: {X_train.shape} \n Shape of y_train: {y_train.shape} \n Shape of X_test: {X_test.shape} \n Shape of y_test: {y_test.shape}')
    return X_train, y_train, X_test, y_test