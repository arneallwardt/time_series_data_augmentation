import torch.nn as nn
import torch.nn.functional as F

class AutoencoderV1(nn.Module):
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


class AutoencoderV2(nn.Module):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        # N, 5, 12
        
        self.conv1 = nn.Conv1d(5, 5, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(5, 5, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv1d(5, 5, kernel_size=4, stride=1, padding=0)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose1d(5, 5, kernel_size=4, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose1d(5, 5, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose1d(5, 5, kernel_size=4, stride=2, padding=1)
    
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


        # Decoder
        x = F.relu(self.deconv1(x))
        print(f'After conv_tran1: {x.shape}') if self.verbose else None

        x = F.relu(self.deconv2(x))
        print(f'After conv_tran2: {x.shape}') if self.verbose else None

        x = F.sigmoid(self.deconv3(x))
        print(f'After conv_tran3: {x.shape}') if self.verbose else None
        
        x = x.permute(0, 2, 1)
        print(f'After re-permute: {x.shape}') if self.verbose else None

        return x