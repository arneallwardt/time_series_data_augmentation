import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy as dc

class ConvAE1(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        # N, 12, 5
        
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(2, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(3, 2), stride=1, padding=0)
        self.conv3 = nn.Conv2d(4, 6, kernel_size=(5, 2), stride=1, padding=0) #n, 6, 1, 1
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(6, 4, kernel_size=(5, 2), stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(4, 2, kernel_size=(3, 2), stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(2, 1, kernel_size=(2, 3), stride=2, padding=(1, 1))
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        print(f'Input after unsqueeze: {x.shape}') if self.verbose else None

        x = F.relu(self.conv1(x))
        print(f'After conv1: {x.shape}') if self.verbose else None

        x = F.relu(self.conv2(x))
        print(f'After conv2: {x.shape}') if self.verbose else None

        x = F.relu(self.conv3(x))
        print(f'After conv3: {x.shape}') if self.verbose else None


        x = F.relu(self.deconv1(x))
        print(f'After conv_tran1: {x.shape}') if self.verbose else None

        x = F.relu(self.deconv2(x))
        print(f'After conv_tran2: {x.shape}') if self.verbose else None

        x = F.sigmoid(self.deconv3(x))
        print(f'After conv_tran3: {x.shape}') if self.verbose else None

        x = x.squeeze(1)
        print(f'After squeeze: {x.shape}') if self.verbose else None
        
        return x


class ConvAE2(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        # N, 5, 12
        
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 5, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(5, 5, kernel_size=4, stride=1, padding=0),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(5, 5, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(5, 5, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(5, 5, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    
    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        x = self.decoder(x)

        x = x.permute(0, 2, 1)
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
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 12, 5)
        return x
    

class ConvAE3(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        self.conv1 = nn.Conv1d(5, 6, kernel_size=8, stride=1, padding=1) # N, 7, 7
        self.conv2 = nn.Conv1d(6, 8, kernel_size=6, stride=1, padding=1) # N, 8, 4
        self.fc1 = nn.Linear(8*4, 4) # N, 4
        
        # Decoder
        self.fc2 = nn.Linear(4, 16) # N, 16
        self.conv_tran1 = nn.ConvTranspose1d(8, 6, kernel_size=6, stride=1, padding=1)
        self.conv_tran2 = nn.ConvTranspose1d(6, 5, kernel_size=6, stride=2, padding=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        print(f'Input after unsqueeze and permute: {x.shape}') if self.verbose else None

        # Encoder
        x = F.relu(self.conv1(x))
        print(f'After conv1: {x.shape}') if self.verbose else None

        x = F.relu(self.conv2(x))
        print(f'After conv2: {x.shape}') if self.verbose else None

        x = F.relu(self.fc1(x.flatten(start_dim=1)))
        print(f'After fc1: {x.shape}') if self.verbose else None


        # Decoder
        x = F.relu(self.fc2(x)).reshape(x.shape[0], 8, 2)
        print(f'After fc2: {x.shape}') if self.verbose else None

        x = x.reshape(-1, 8, 2)
        print(f'After reshape: {x.shape}') if self.verbose else None

        x = F.relu(self.conv_tran1(x))
        print(f'After conv_tran1: {x.shape}') if self.verbose else None

        x = F.relu(self.conv_tran2(x))
        print(f'After conv_tran2: {x.shape}') if self.verbose else None
        
        x = x.permute(0, 2, 1)
        print(f'After re-permute: {x.shape}') if self.verbose else None

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

            X_train_hat_copy[:, :, 0] *= 2
            X_train_hat_copy[:, :, 1] *= 2
            X_train_copy[:, :, 0] *= 2
            X_train_copy[:, :, 1] *= 2

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