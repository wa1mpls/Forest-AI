import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, FEATURES
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import tensorflow as tf
from pathlib import Path
import rasterio
from sklearn.model_selection import train_test_split

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

class ForestDataset:
    def __init__(self, sentinel_dir, gedi_dir, config):
        """
        Initialize dataset with paths and configuration
        
        Args:
            sentinel_dir (Path): Directory containing Sentinel-2 images
            gedi_dir (Path): Directory containing GEDI data
            config (dict): Configuration dictionary containing preprocessing settings
        """
        self.sentinel_dir = Path(sentinel_dir)
        self.gedi_dir = Path(gedi_dir)
        self.config = config
        
        # Load and preprocess data
        self.sentinel_data = self._load_sentinel_data()
        self.gedi_data = self._load_gedi_data()
        
    def _load_sentinel_data(self):
        """Load and preprocess Sentinel-2 data"""
        sentinel_data = {}
        for band in self.config["sentinel_bands"]:
            band_path = self.sentinel_dir / f"{band}.tif"
            if band_path.exists():
                with rasterio.open(band_path) as src:
                    data = src.read(1)
                    if self.config["normalize"]:
                        data = (data - np.min(data)) / (np.max(data) - np.min(data))
                    sentinel_data[band] = data
                    
        # Compute NDVI if requested
        if self.config["compute_ndvi"] and "B4" in sentinel_data and "B8" in sentinel_data:
            ndvi = (sentinel_data["B8"] - sentinel_data["B4"]) / (sentinel_data["B8"] + sentinel_data["B4"])
            sentinel_data["NDVI"] = ndvi
            
        return sentinel_data
    
    def _load_gedi_data(self):
        """Load and preprocess GEDI data"""
        gedi_files = list(self.gedi_dir.glob("*.csv"))
        if not gedi_files:
            raise FileNotFoundError(f"No GEDI data found in {self.gedi_dir}")
            
        gedi_data = pd.concat([pd.read_csv(f) for f in gedi_files])
        return gedi_data
    
    def split_data(self, train_ratio, val_ratio, test_ratio, random_seed=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            test_ratio (float): Ratio of test data
            random_seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        # Combine Sentinel and GEDI data
        X = np.stack([self.sentinel_data[band] for band in self.config["sentinel_bands"]], axis=-1)
        y = self.gedi_data["agbd"].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1-train_ratio, random_state=random_seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_size, random_state=random_seed
        )
        
        # Create TensorFlow datasets
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        return train_data, val_data, test_data 