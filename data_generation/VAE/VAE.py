import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy as dc


class ConvVAE(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
    
        self.common_conv = nn.Sequential(
            nn.Conv1d(5, 7, kernel_size=7, stride=1, padding=0),
            nn.ReLU()
        )
        
        self.mean_conv = nn.Sequential(
            nn.Conv1d(7, 10, kernel_size=6, stride=1, padding=0),
            nn.ReLU()
        )

        self.log_var_conv = nn.Sequential(
            nn.Conv1d(7, 10, kernel_size=6, stride=1, padding=0),
            nn.ReLU()
        )

        self.mean_fc = nn.Sequential(
            nn.Linear(10, 4),
            nn.ReLU()
        )
        
        self.log_var_fc = nn.Sequential(
            nn.Linear(10, 4),
            nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(8, 6, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
        )

        self.conv_tran_1 = nn.ConvTranspose1d(8, 6, kernel_size=4, stride=2, padding=0)
        self.conv_tran_2 = nn.ConvTranspose1d(6, 5, kernel_size=7, stride=1, padding=0)
            
    
    def encode(self, x):
        x = x.permute(0, 2, 1)
        print(f'Input after unsqueeze and permute: {x.shape}') if self.verbose else None

        out = self.common_conv(x)
        print(f'After common_conv: {out.shape}') if self.verbose else None

        mean = self.mean_conv(out)
        print(f'After mean_conv: {mean.shape}') if self.verbose else None
        mean = self.mean_fc(torch.flatten(mean, start_dim=1))
        print(f'After mean_fc: {mean.shape}') if self.verbose else None

        log_var = self.log_var_conv(out)
        print(f'After log_var_conv: {log_var.shape}') if self.verbose else None
        log_var = self.log_var_fc(torch.flatten(log_var, start_dim=1))
        print(f'After log_var_fc: {log_var.shape}') if self.verbose else None


        return mean, log_var
    
    def sample(self, mean, log_var):
        std = torch.exp(0.5*log_var) # get standard deviation

        z = torch.randn_like(std) # sample from normal distribution
        z = z * std + mean # reparameterization trick

        return z
    
    def decode(self, x):
        # out = self.decoder_conv(z.unsqueeze(1))
        # print(f'After decoder_conv: {out.shape}') if self.verbose else None

        out = self.decoder_fc(x) # shape (n, 16)
        print(f'After decoder_fc: {out.shape}') if self.verbose else None

        out = out.reshape((out.size(0), 8, 2)) # shape (n, 8, 2)
        print(f'After reshape: {out.shape}') if self.verbose else None

        out = F.relu(self.conv_tran_1(out))
        print(f'After decoder_conv_1: {out.shape}') if self.verbose else None
        out = F.relu(self.conv_tran_2(out))
        print(f'After decoder_conv_2: {out.shape}') if self.verbose else None

        out = out.permute(0, 2, 1)
        print(f'After re-permute: {out.shape}') if self.verbose else None
        return out

    def forward(self, x):
        # Encoder
        mean, log_var = self.encode(x)
        
        # Sampling
        z = self.sample(mean, log_var)

        # Decoder 
        out = self.decode(z)
        
        return mean, log_var, out

class FCVAE(nn.Module):
    # Code adapted from https://github.com/pytorch/examples/blob/main/vae/main.py
    def __init__(self):
        super(FCVAE, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(5*12, 32),
            nn.ReLU()
        )

        self.fc21 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.fc22 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 5*12),
            nn.ReLU()
        )

    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1) # return mu and logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # get standard deviation
        eps = torch.randn_like(std) # sample from normal distribution
        return mu + eps*std # reparameterization trick
    
    def decode(self, z):
        h3 = self.fc3(z)
        h3 = h3.reshape((z.size(0), 12, 5))
        return h3
    
    def forward(self, x):
        mu, logvar = self.encode(torch.flatten(x, start_dim=1)) # Encoder
        z = self.reparameterize(mu, logvar) # Sampling and reparameterization
        return mu, logvar, self.decode(z)


class LSTMVAE(nn.Module):

    def __init__(self):
        super(LSTMVAE, self).__init__()

        self.input_size = 5
        self.hidden_size = 6
        self.num_layers = 1
        self.latent_size = 16
        self.latent_size_2 = 4
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.latent_size)

        self.mean_fc = nn.Linear(self.latent_size, self.latent_size_2)
        self.log_var_fc = nn.Linear(self.latent_size, self.latent_size_2)

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_size_2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 5*12),
            nn.ReLU()
        )

    def common_lstm(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc(out[:, -1, :]))

        return out
    
    def encode(self, x):
        out = self.common_lstm(x)
        
        # get mean and log_var
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)

        return mean, log_var
    
    def sample(self, mean, log_var):
        std = torch.exp(0.5*log_var) # get standard deviation

        z = torch.randn_like(std) # sample from normal distribution
        z = z * std + mean # reparameterization trick

        return z
    
    def decode(self, z):
        out = self.decoder_fc(z)
        out = out.reshape((z.size(0), 12, 5))
        return out
    
    def forward(self, x):
        # Encoder 
        mean, log_var = self.encode(x)

        # Sampling
        z = self.sample(mean, log_var)

        # Decoder 
        out = self.decode(z)

        return mean, log_var, out

    

def train_vae(model, 
              hyperparameters, 
              train_loader, 
              val_loader, 
              criterion, 
              optimizer,
              model_name,
              patience=10):
    
    best_val_loss = np.inf
    num_epochs_no_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(hyperparameters["num_epochs"]):

        ### Training ###        
        accumulated_recon_loss = 0
        accumulated_kl_loss = 0
        accumulated_loss = 0

        for X_train, _ in train_loader: 
            
            X_train = X_train.float().to(hyperparameters["device"])
        
            ### Training
            model.train()

            mean, log_var, out = model(X_train)

            # calculate loss
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var, dim=-1))
            reconstruction_loss = criterion(out, X_train)
            loss = reconstruction_loss + 0.001 * kl_loss # add weight to kl_loss, because it is not as important as recon loss
            
            # save losses for later
            accumulated_recon_loss += reconstruction_loss.item()
            accumulated_kl_loss += kl_loss.item()
            accumulated_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ### Validation
        accumulated_val_loss = 0
        model.eval()
        with torch.inference_mode():
            for X_val, _ in val_loader:
                X_val = X_val.float().to(hyperparameters["device"])

                val_mean, val_log_var, val_out = model(X_val)

                val_kl_loss = torch.mean(0.5 * torch.sum(torch.exp(val_log_var) + val_mean**2 - 1 - val_log_var, dim=-1))
                val_reconstruction_loss = criterion(val_out, X_val)
                val_loss = val_reconstruction_loss + 0.001 * val_kl_loss

                accumulated_val_loss += (val_loss.item())

            
            # Check for early stopping
            avg_val_loss_accross_batches = accumulated_val_loss / len(val_loader)
            if avg_val_loss_accross_batches < best_val_loss:

                # reset early stopping counter and save best validation loss
                best_val_loss = avg_val_loss_accross_batches
                num_epochs_no_improvement = 0

                # save model
                torch.save(model.state_dict(), f"{model_name}_checkpoint.pth")

            else:
                print(f'INFO: Validation loss did not improve in epoch {epoch + 1}')
                num_epochs_no_improvement += 1
            

        ### Logging ###

        train_losses.append(accumulated_loss / len(train_loader))
        val_losses.append(avg_val_loss_accross_batches)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1} | avg. Recon Loss: {(accumulated_recon_loss / len(train_loader)):.4f} | avg. KL Loss: {(accumulated_kl_loss / len(train_loader)):.4f} | avg. Train Loss: {(accumulated_loss / len(train_loader)):.4f} | avg. Val Loss: {avg_val_loss_accross_batches:.4f}")

        if num_epochs_no_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return train_losses, val_losses