import torch
import torch.nn as nn
import timm
from utils.logger import get_logger

logger = get_logger(__name__)

class VisionTransformer(nn.Module):
    def __init__(self, config):
        """Initialize Vision Transformer model"""
        super().__init__()
        self.config = config
        
        # Load pretrained ViT
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Add regression head
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Freeze early layers
        for param in list(self.vit.parameters())[:-6]:
            param.requires_grad = False
    
    def forward(self, x):
        """Forward pass"""
        # Get features from ViT
        features = self.vit(x)
        
        # Apply regression head
        output = self.regression_head(features)
        
        return output.squeeze()
    
    def train_step(self, batch, optimizer, criterion):
        """Single training step"""
        # Get batch data
        images, targets = batch
        
        # Forward pass
        predictions = self(images)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch, criterion):
        """Single validation step"""
        # Get batch data
        images, targets = batch
        
        # Forward pass
        with torch.no_grad():
            predictions = self(images)
        
        # Calculate loss
        loss = criterion(predictions, targets)
        
        return loss.item(), predictions, targets
    
    def predict(self, images):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            predictions = self(images)
        return predictions
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
    
    def get_attention_maps(self, images):
        """Get attention maps from the model"""
        self.eval()
        with torch.no_grad():
            # Get attention weights from the last layer
            attention_weights = self.vit.get_attention_weights(images)
        return attention_weights 