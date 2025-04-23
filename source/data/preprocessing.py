import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..config import FEATURES
import os
from pathlib import Path
import rasterio
from rasterio.windows import Window

def clean_data(df):
    """Clean and preprocess the data"""
    # Convert all features to numeric
    all_features = []
    for features in FEATURES.values():
        all_features.extend(features)
    
    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=all_features)
    
    return df

def normalize_data(df):
    """Normalize the data using MinMaxScaler"""
    all_features = []
    for features in FEATURES.values():
        all_features.extend(features)
    
    scaler = MinMaxScaler()
    df[all_features] = scaler.fit_transform(df[all_features])
    
    return df, scaler

def create_quality_mask(df):
    """Create quality mask based on GEDI quality flags"""
    # Filter based on quality flags
    quality_mask = (
        (df['l2_quality_flag'] == 1) &  # Good quality
        (df['degrade_flag'] == 0) &     # Not degraded
        (df['sensitivity'] > 0.95)      # High sensitivity
    )
    
    return quality_mask

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train, validation and test sets"""
    # Shuffle the data
    df = df.sample(frac=1, random_state=42)
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split the data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def prepare_data(csv_path, output_dir):
    """Prepare data for training"""
    # Read and clean data
    df = pd.read_csv(csv_path)
    df = clean_data(df)
    
    # Create quality mask
    quality_mask = create_quality_mask(df)
    df = df[quality_mask]
    
    # Normalize data
    df, scaler = normalize_data(df)
    
    # Save scaler
    np.save(os.path.join(output_dir, 'scaler.npy'), scaler)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    return train_df, val_df, test_df

def normalize_bands(band_data):
    """
    Normalize band data to [0, 1] range
    
    Args:
        band_data (np.ndarray): Input band data
        
    Returns:
        np.ndarray: Normalized band data
    """
    return (band_data - np.min(band_data)) / (np.max(band_data) - np.min(band_data))

def compute_ndvi(red_band, nir_band):
    """
    Compute NDVI from red and NIR bands
    
    Args:
        red_band (np.ndarray): Red band data
        nir_band (np.ndarray): NIR band data
        
    Returns:
        np.ndarray: NDVI values
    """
    return (nir_band - red_band) / (nir_band + red_band + 1e-8)

def extract_patches(image, patch_size, stride):
    """
    Extract patches from an image
    
    Args:
        image (np.ndarray): Input image
        patch_size (tuple): Size of patches (height, width)
        stride (int): Stride for patch extraction
        
    Returns:
        np.ndarray: Extracted patches
    """
    patches = []
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    for y in range(0, height - patch_height + 1, stride):
        for x in range(0, width - patch_width + 1, stride):
            patch = image[y:y+patch_height, x:x+patch_width]
            patches.append(patch)
            
    return np.array(patches)

def preprocess_gedi_data(gedi_df, features):
    """
    Preprocess GEDI data
    
    Args:
        gedi_df (pd.DataFrame): GEDI data
        features (list): List of features to use
        
    Returns:
        pd.DataFrame: Preprocessed GEDI data
    """
    # Select features
    gedi_df = gedi_df[features].copy()
    
    # Remove outliers
    for col in gedi_df.columns:
        q1 = gedi_df[col].quantile(0.25)
        q3 = gedi_df[col].quantile(0.75)
        iqr = q3 - q1
        gedi_df = gedi_df[
            (gedi_df[col] >= q1 - 1.5 * iqr) & 
            (gedi_df[col] <= q3 + 1.5 * iqr)
        ]
    
    # Normalize features
    scaler = StandardScaler()
    gedi_df[features] = scaler.fit_transform(gedi_df[features])
    
    return gedi_df

def create_training_patches(sentinel_path, gedi_df, patch_size, stride):
    """
    Create training patches from Sentinel-2 and GEDI data
    
    Args:
        sentinel_path (Path): Path to Sentinel-2 image
        gedi_df (pd.DataFrame): GEDI data
        patch_size (tuple): Size of patches
        stride (int): Stride for patch extraction
        
    Returns:
        tuple: (patches, labels)
    """
    with rasterio.open(sentinel_path) as src:
        # Read all bands
        bands = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            bands.append(band)
        image = np.stack(bands, axis=-1)
    
    # Extract patches
    patches = extract_patches(image, patch_size, stride)
    
    # Match patches with GEDI data
    labels = []
    for patch in patches:
        # Find corresponding GEDI data (simplified)
        # In practice, you would need to implement proper spatial matching
        label = gedi_df.sample(1)["agbd"].values[0]
        labels.append(label)
    
    return np.array(patches), np.array(labels) 