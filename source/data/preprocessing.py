import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ..config import FEATURES
import os

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