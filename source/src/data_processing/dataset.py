import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Tuple, Dict, List
import logging
import json
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForestDataset(Dataset):
    def __init__(self, config_path: str = "configs/data_config.yaml", split: str = "train"):
        """
        Initialize Forest Dataset
        
        Args:
            config_path: Path to configuration file
            split: Data split (train/val/test)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.patches_dir = Path(self.config['paths']['patches_dir'])
        self.labels_dir = Path(self.config['paths']['labels_dir'])
        
        # Load data
        self.patches, self.labels = self._load_data(split)
        
        # Get preprocessing parameters
        self.normalize = self.config['preprocessing']['normalize']
        self.image_size = self.config['preprocessing']['image_size']
        
    def _load_data(self, split: str) -> Tuple[List[str], List[float]]:
        """
        Load patches and labels for specified split
        
        Args:
            split: Data split (train/val/test)
            
        Returns:
            Tuple[List[str], List[float]]: List of patch paths and labels
        """
        # Load matched data
        matched_path = self.labels_dir / 'matched_data.csv'
        if not matched_path.exists():
            raise FileNotFoundError(f"Matched data not found at {matched_path}")
        
        df = pd.read_csv(matched_path)
        
        # Split data
        train_df, temp_df = train_test_split(
            df,
            test_size=1-self.config['split']['train_ratio'],
            random_state=self.config['split']['random_seed']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=self.config['split']['test_ratio']/(1-self.config['split']['train_ratio']),
            random_state=self.config['split']['random_seed']
        )
        
        # Get data for specified split
        if split == "train":
            split_df = train_df
        elif split == "val":
            split_df = val_df
        elif split == "test":
            split_df = test_df
        else:
            raise ValueError(f"Invalid split: {split}")
        
        return split_df['path'].tolist(), split_df['agbd'].tolist()
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Patch and label
        """
        # Load patch
        patch_path = self.patches[idx]
        patch = np.load(patch_path)
        
        # Convert to tensor
        patch = torch.from_numpy(patch).float()
        
        # Normalize if required
        if self.normalize:
            patch = (patch - patch.mean()) / patch.std()
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return patch, label

def get_dataloaders(
    config_path: str = "configs/data_config.yaml",
    batch_size: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for train, validation and test sets
    
    Args:
        config_path: Path to configuration file
        batch_size: Batch size (default from config)
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get batch size
    if batch_size is None:
        batch_size = config['preprocessing']['batch_size']
    
    # Create datasets
    train_dataset = ForestDataset(config_path, split="train")
    val_dataset = ForestDataset(config_path, split="val")
    test_dataset = ForestDataset(config_path, split="test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

class DataProcessor:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize data processor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.patches_dir = Path(self.config['paths']['patches_dir'])
        self.labels_dir = Path(self.config['paths']['labels_dir'])
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def process_data(self, sentinel_path: str, gedi_path: str) -> None:
        """
        Process Sentinel-2 and GEDI data
        
        Args:
            sentinel_path: Path to Sentinel-2 patches metadata
            gedi_path: Path to GEDI data
        """
        # Load GEDI processor
        from src.data_processing.gedi import GEDIProcessor
        gedi_processor = GEDIProcessor()
        
        # Match GEDI points with Sentinel patches
        matched_data = gedi_processor.match_with_sentinel(gedi_path, sentinel_path)
        
        # Save matched data
        output_path = self.labels_dir / 'matched_data.csv'
        matched_data.to_csv(output_path, index=False)
        
        logger.info(f"Processed {len(matched_data)} matched samples")
        
    def create_dataset(self) -> None:
        """
        Create PyTorch dataset from processed data
        """
        # Get dataloaders
        train_loader, val_loader, test_loader = get_dataloaders()
        
        # Save dataset statistics
        stats = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'total_samples': len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        }
        
        stats_path = self.labels_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        logger.info(f"Created dataset with {stats['total_samples']} samples")
        logger.info(f"Train: {stats['train_samples']}, Val: {stats['val_samples']}, Test: {stats['test_samples']}") 