import torch
import torch.nn as nn

class EnhancedFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        # Define vegetation indices
        self.ndvi = lambda x: (x[:,3] - x[:,2]) / (x[:,3] + x[:,2] + 1e-6)
        self.evi = lambda x: 2.5 * (x[:,3] - x[:,2]) / (x[:,3] + 6*x[:,2] - 7.5*x[:,0] + 1)
        self.ndwi = lambda x: (x[:,1] - x[:,3]) / (x[:,1] + x[:,3] + 1e-6)
        
    def forward(self, x):
        # x: [batch, channels, height, width]
        # Calculate vegetation indices
        ndvi = self.ndvi(x)
        evi = self.evi(x)
        ndwi = self.ndwi(x)
        
        # Stack features
        enhanced = torch.cat([
            x,
            ndvi.unsqueeze(1),
            evi.unsqueeze(1),
            ndwi.unsqueeze(1)
        ], dim=1)
        
        return enhanced 