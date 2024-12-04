# CSE572-Final_Project
This repository is built concerning the CSE 572 final project source code.
## Getting Started 
This model using pytorch and convkan 
for details find the information [here](https://pypi.org/project/convkan/)
ConvKans built using EfficientKANs
```bash
pip install convkan
```

### AutoEncoder Architecture 

### Our AutoEncoder Architecture
Changed the Convolutional Layers to ConvKANs. 
```bash
import torch
import torch.nn as nn
from convkan import ConvKAN, LayerNorm2D

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            ConvKAN(in_channels, 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2D(4),
            nn.LeakyReLU(0.2, inplace=True),
            ConvKAN(4, 8, kernel_size=3, stride=2, padding=1),
            LayerNorm2D(8),
            nn.LeakyReLU(0.2, inplace=True),
            ConvKAN(8, 16, kernel_size=3, stride=2, padding=1),
            LayerNorm2D(16),
            nn.LeakyReLU(0.2, inplace=True),
            ConvKAN(16, 16, kernel_size=3, stride=2, padding=1),
            LayerNorm2D(16),
            nn.LeakyReLU(0.2, inplace=True),
            ConvKAN(16, 32, kernel_size=3, stride=2, padding=1),
            LayerNorm2D(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.flatten = nn.Flatten()
        self.feature_size = 32 * 4 * 4  # 32 channels, 7x7 feature map
        self.fc = nn.Linear(self.feature_size, latent_dim)
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (32, 4, 4))
        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4 -> 8
            ConvKAN(32, 16, kernel_size=3, stride=1, padding=1),
            LayerNorm2D(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8 -> 16
            ConvKAN(16, 8, kernel_size=3, stride=1, padding=1),
            LayerNorm2D(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16 -> 32
            ConvKAN(8, 8, kernel_size=3, stride=1, padding=1),
            LayerNorm2D(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32 -> 64
            ConvKAN(8, 4, kernel_size=3, stride=1, padding=1),
            LayerNorm2D(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=1.75, mode='bilinear', align_corners=True),  # Fine-tune output size
            ConvKAN(4, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.deconv_layers(x)
        return x

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
```
## Output of the AutoEncoder
