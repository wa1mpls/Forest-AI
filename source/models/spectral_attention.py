import torch
import torch.nn as nn

class SpectralAttention(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_bands, num_bands//2),
            nn.ReLU(),
            nn.Linear(num_bands//2, num_bands),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        attention = self.attention(x.mean(dim=[2,3]))
        return x * attention.unsqueeze(-1).unsqueeze(-1) 