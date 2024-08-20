import os
import sys 

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import numpy as np
from utilities import accuracy
from copy import deepcopy as dc


class LSTMRegression(nn.Module):
    def __init__(self, device, input_size=1, hidden_size=4, num_stacked_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.num_directions = 2 if bidirectional else 1
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, bidirectional=bidirectional) # already includes activation layers
        self.fc = nn.Linear(hidden_size*self.num_directions, 1) # fully connected layer with output = 1

    def forward(self, x):
        '''
        Args:
            x: torch.Tensor with shape (batch_size, seq_len, input_size) (if batch_first=True, else (seq_len, batch_size, input_size))

        Returns:
            out: torch.Tensor with shape (batch_size, 1)
        '''

        batch_size = x.size(0) # get batch size bc input size is 1

        h0 = torch.zeros(self.num_stacked_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0)) 

        out = self.fc(out[:, -1, :]) # run output through fully connected layer
        return out
    


### TRAIN AND TEST LOOP ###

def train_one_epoch(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        device):
    
    model.train()
    running_train_loss = 0.0
    running_train_acc = 0.0

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)  
        
        optimizer.zero_grad()

        train_logits = model(x_batch)

        train_loss = criterion(train_logits, y_batch)
        running_train_loss += train_loss.item()
        running_train_acc += accuracy(y_true=y_batch, y_pred=torch.round(torch.sigmoid(train_logits)))

        train_loss.backward(retain_graph=True) # retain graph to access the current hidden state in the next forward pass

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients to prevent exploding gradients

        optimizer.step()

    avg_train_loss = running_train_loss / len(train_loader)
    avg_train_acc = running_train_acc / len(train_loader)
    return avg_train_loss, avg_train_acc


def validate_one_epoch(
        model, 
        val_loader, 
        criterion, 
        device):
    
    '''Validates the model and returns the average validation loss.'''
    
    model.eval()
    running_test_loss = 0.0
    running_test_acc = 0.0

    with torch.inference_mode():
        for _, batch in enumerate(val_loader):
            x_batch, y_batch = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            test_pred = model(x_batch) # output in logits

            test_loss = criterion(test_pred, y_batch)
            test_acc = accuracy(y_true=y_batch, y_pred=torch.round(torch.sigmoid(test_pred)))
            
            running_test_acc += test_acc
            running_test_loss += test_loss.item()

    # log validation loss
    avg_test_loss_across_batches = running_test_loss / len(val_loader)
    avg_test_acc_accross_batches = running_test_acc / len(val_loader)
    return avg_test_loss_across_batches, avg_test_acc_accross_batches


def train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device,
        save_path,
        verbose=True,
        patience=5, 
        num_epochs=1000):
    
    '''Trains the model and returns the best validation loss aswell as the trained model. Stops training if the validation loss does not improve for patience epochs.'''

    train_losses = []    
    train_accs = []
    val_losses = []    
    val_accs = []
    best_validation_loss = np.inf
    num_epoch_without_improvement = 0
    for epoch in range(num_epochs):
        current_train_loss, current_train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        current_validation_loss, current_validation_acc = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(current_train_loss)
        train_accs.append(current_train_acc)
        val_losses.append(current_validation_loss)
        val_accs.append(current_validation_acc)
        
        # early stopping
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            num_epoch_without_improvement = 0
            torch.save(model.state_dict(), save_path) # save best model to use for testing
        else:
            print(f'INFO: Validation loss did not improve in epoch {epoch + 1}') if verbose else None
            num_epoch_without_improvement += 1

        if num_epoch_without_improvement >= patience:
            print(f'Early stopping after {epoch + 1} epochs') if verbose else None
            break

        if epoch % 10 == 0:
            print(f'Epoch: {epoch + 1}') if verbose else None
            print(f'Train Loss: {current_train_loss} // Train Acc: {current_train_acc}') if verbose else None
            print(f'Val Loss: {current_validation_loss} // Val Acc: {current_validation_acc}') if verbose else None
            print(f'*' * 50) if verbose else None

    return train_losses, train_accs, val_losses, val_accs, model