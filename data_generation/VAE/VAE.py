import torch.nn as nn
import numpy as np
import torch


class FCVAE(nn.Module):
    # Code adapted from https://github.com/pytorch/examples/blob/main/vae/main.py
    def __init__(self, latent_size_1=32, latent_size_2=8):
        super(FCVAE, self).__init__()
        self.latent_size_1 = latent_size_1
        self.latent_size_2 = latent_size_2

        self.fc1 = nn.Sequential(
            nn.Linear(5*12, self.latent_size_1),
            nn.ReLU()
        )

        self.fc21 = nn.Sequential(
            nn.Linear(self.latent_size_1, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_size_2)
        )

        self.fc22 = nn.Sequential(
            nn.Linear(self.latent_size_1, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_size_2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_size_2, 16),
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
            loss = reconstruction_loss + 0.002 * kl_loss # add weight to kl_loss, because it is not as important as recon loss
            
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