import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 32 -> 16
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 128 -> 256
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
