import torch
import torch.nn as nn
from transformers import ViTModel
from .spectral_attention import SpectralAttention
from .enhanced_features import EnhancedFeatures
from ..config import MODEL_CONFIG

class HybridForestModel(nn.Module):
    def __init__(self, num_outputs=None):
        super().__init__()
        if num_outputs is None:
            num_outputs = MODEL_CONFIG['num_outputs']
        
        # CNN for low-level features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # ViT for long-range interactions
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Spectral attention
        self.spectral_attention = SpectralAttention(num_bands=7)  # 3 RGB + NDVI + EVI + NDWI
        
        # Enhanced features
        self.enhanced_features = EnhancedFeatures()
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128*56*56, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        # CNN features
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # ViT features
        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        
        # Enhanced features
        enhanced = self.enhanced_features(x)
        enhanced = self.spectral_attention(enhanced)
        enhanced = enhanced.mean(dim=[2,3])
        
        # Fusion
        combined = torch.cat([cnn_features, vit_features, enhanced], dim=1)
        features = self.fusion(combined)
        
        # Regression
        output = self.regressor(features)
        
        return output 