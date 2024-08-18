import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy as dc    


class LSTMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 5
        self.hidden_size = 4
        self.num_layers = 1
        self.latent_size = 8
        
        self.lstm_encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_encoder = nn.Linear(self.hidden_size, self.latent_size)
        
        self.lstm_decoder = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_decoder = nn.Linear(self.hidden_size, 12*5)

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 48),
            nn.ReLU(),
            nn.Linear(48, 12*5),
            nn.ReLU()
        )

    def encode(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        encoded, _ = self.lstm_encoder(x, (h0, c0))
        encoded = self.fc_encoder(encoded[:, -1, :])

        return encoded
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decoder(x)
        x = x.view(-1, 12, 5)
        return x


class FCAE(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        # N, 12, 5
        
        self.encoder = nn.Sequential(
            nn.Linear(12*5, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 12, 5)
        return x
    

def train_autoencoder(model, 
              hyperparameters, 
              train_loader, 
              val_loader, 
              criterion, 
              optimizer,
              save_path,
              patience=10):
    
    best_val_loss = np.inf
    num_epochs_no_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(hyperparameters['num_epochs'])):

        ### Training ###
        accumulated_train_loss = 0

        for _, (X_train, _) in enumerate(train_loader):

            model.train()
            X_train = X_train.float().to(hyperparameters['device'])

            X_train_hat = model(X_train)

            # weigh first 2 features higher
            X_train_hat_copy = X_train_hat.clone()
            X_train_copy = X_train.clone()

            X_train_hat_copy[:, :, 0] *= 1.5
            X_train_hat_copy[:, :, 1] *= 1.5
            X_train_copy[:, :, 0] *= 1.5
            X_train_copy[:, :, 1] *= 1.5

            train_loss = criterion(X_train_hat, X_train)
            accumulated_train_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        avg_train_loss_accross_batches = accumulated_train_loss / len(train_loader)


        ### Validation ###
        accumulated_val_loss = 0
        model.eval()
        with torch.inference_mode():

            for _, (X_val, _) in enumerate(val_loader):
                X_val = X_val.float().to(hyperparameters['device'])

                X_val_hat = model(X_val)

                # weigh first 2 features higher
                X_val_hat_copy = X_val_hat.clone()
                X_val_copy = X_val.clone()

                X_val_hat_copy[:, :, 0] *= 2
                X_val_hat_copy[:, :, 1] *= 2
                X_val_copy[:, :, 0] *= 2
                X_val_copy[:, :, 1] *= 2

                val_loss = criterion(X_val_hat, X_val)
                accumulated_val_loss += val_loss.item()


            # Check for early stopping
            avg_val_loss_accross_batches = accumulated_val_loss / len(val_loader)
            if avg_val_loss_accross_batches < best_val_loss:
                best_val_loss = avg_val_loss_accross_batches
                num_epochs_no_improvement = 0
                torch.save(model.state_dict(), save_path) # save best model to use for testing

            else:
                print(f'INFO: Validation loss did not improve in epoch {epoch + 1}')
                num_epochs_no_improvement += 1


        ### Logging and Plotting ###

        train_losses.append(avg_train_loss_accross_batches)
        val_losses.append(avg_val_loss_accross_batches)

        print(f'Epoch: {epoch} \n\b Train Loss: {avg_train_loss_accross_batches} \n\b Val Loss: {avg_val_loss_accross_batches}')
        print('*' * 50)

        if epoch % 10 == 0:
            plot_train = dc(X_train.detach().numpy())
            plot_hat = dc(X_train_hat.detach().numpy())

            plt.figure(figsize=(10, 5))
            plt.title(f'Epoch {epoch} | Original vs Synthetic')
            plt.plot(plot_train[0, :, 0], label='Original')
            plt.plot(plot_hat[0, :, 0], label='Synthetic')
            plt.legend()

        if num_epochs_no_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return train_losses, val_losses