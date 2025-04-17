import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from ..config import DATA_CONFIG, FEATURES
import pandas as pd
from torch.utils.data import DataLoader

class ForestDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform or self._get_default_transform()
        
        # Get all features
        self.features = []
        for features in FEATURES.values():
            self.features.extend(features)
        
        # Convert to numeric
        for col in self.features:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Drop missing values
        self.data = self.data.dropna(subset=self.features)
        
        # Limit number of samples
        self.num_images = min(DATA_CONFIG['max_samples'], len(self.data))
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize(DATA_CONFIG['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # Get image path
        image_path = os.path.join(self.image_folder, f"image_{idx}.png")
        metadata_path = image_path.replace('.png', '_metadata.json')
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create label from metadata
        label = torch.tensor([metadata[feat] for feat in self.features], dtype=torch.float32)
        
        return image, label

def get_dataloaders(train_df, val_df, test_df, image_folder, batch_size=None):
    """Create dataloaders for train, validation and test sets"""
    if batch_size is None:
        batch_size = DATA_CONFIG['batch_size']
    
    # Create datasets
    train_dataset = ForestDataset(train_df, image_folder)
    val_dataset = ForestDataset(val_df, image_folder)
    test_dataset = ForestDataset(test_df, image_folder)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader 