import os
import yaml
import numpy as np
import pandas as pd
import ee
import geemap
from pathlib import Path
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_sentinel_features(sentinel, region, config):
    """Extract features from Sentinel-2 data"""
    logger.info("Extracting Sentinel-2 features...")
    
    # Get image
    image = sentinel.clip(region)
    
    # Calculate spectral indices
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # Stack all features
    features = image.select(config["sentinel"]["bands"]).addBands([ndvi, gndvi, nbr])
    
    return features

def get_gedi_features(gedi_l4a, gedi_l2a, gedi_l2b, region, config):
    """Extract features from GEDI data"""
    logger.info("Extracting GEDI features...")
    
    # Get GEDI data
    gedi_l4a = gedi_l4a.clip(region)
    gedi_l2a = gedi_l2a.clip(region)
    gedi_l2b = gedi_l2b.clip(region)
    
    # Filter by quality criteria
    gedi_l4a = gedi_l4a.filter(ee.Filter.gt('l2_quality_flag', config["gedi"]["filters"]["quality_flag"]))
    gedi_l4a = gedi_l4a.filter(ee.Filter.gt('rh100', config["gedi"]["filters"]["min_rh100"]))
    gedi_l4a = gedi_l4a.filter(ee.Filter.lt('slope', config["gedi"]["filters"]["max_slope"]))
    gedi_l4a = gedi_l4a.filter(ee.Filter.gt('confidence', config["gedi"]["filters"]["confidence"]))
    
    # Combine features
    gedi_features = gedi_l4a.select(config["gedi"]["features"]["l4a"])
    gedi_features = gedi_features.addBands(gedi_l2a.select(config["gedi"]["features"]["l2a"]))
    gedi_features = gedi_features.addBands(gedi_l2b.select(config["gedi"]["features"]["l2b"]))
    
    return gedi_features

def create_patches(features, gedi_features, config):
    """Create patches and assign GEDI labels"""
    logger.info("Creating patches...")
    
    patch_size = config["preprocessing"]["image_size"][0]
    num_features = config["preprocessing"]["num_features"]
    
    # Get image dimensions
    image_info = features.getInfo()
    height = image_info['bands'][0]['dimensions'][0]
    width = image_info['bands'][0]['dimensions'][1]
    
    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    patches = []
    labels = []
    patch_info = []
    
    for i in tqdm(range(num_patches_h)):
        for j in range(num_patches_w):
            # Extract patch
            patch = features.sample(
                region=ee.Geometry.Rectangle([
                    j*patch_size, i*patch_size,
                    (j+1)*patch_size, (i+1)*patch_size
                ]),
                scale=10
            )
            
            # Get GEDI points in this patch
            gedi_points = gedi_features.sample(
                region=ee.Geometry.Rectangle([
                    j*patch_size, i*patch_size,
                    (j+1)*patch_size, (i+1)*patch_size
                ]),
                scale=10
            )
            
            # Get patch data
            patch_data = patch.getInfo()
            gedi_data = gedi_points.getInfo()
            
            if len(gedi_data['features']) >= config["preprocessing"]["min_gedi_points"]:
                # Extract features
                patch_features = np.array([f['properties'] for f in patch_data['features']])
                patch_features = patch_features.reshape(patch_size, patch_size, -1)
                
                # Extract GEDI labels
                gedi_labels = np.array([f['properties']['agbd'] for f in gedi_data['features']])
                
                patches.append(patch_features)
                labels.append(np.mean(gedi_labels))
                
                # Save patch information
                patch_info.append({
                    'patch_id': len(patches) - 1,
                    'x': j*patch_size,
                    'y': i*patch_size,
                    'num_gedi_points': len(gedi_data['features']),
                    'mean_agbd': np.mean(gedi_labels),
                    'std_agbd': np.std(gedi_labels)
                })
    
    return np.array(patches), np.array(labels), pd.DataFrame(patch_info)

def main():
    """Main preprocessing function"""
    # Load config
    config = load_config()
    
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        logger.error("Please authenticate with Google Earth Engine first:")
        logger.error("1. Go to https://earthengine.google.com/")
        logger.error("2. Sign in with your Google account")
        logger.error("3. Run: earthengine authenticate")
        return
    
    # Get data from Google Earth Engine
    from download_dataset import get_sentinel_data, get_gedi_data
    
    sentinel, region = get_sentinel_data(config)
    gedi_l4a, gedi_l2a, gedi_l2b, region = get_gedi_data(config)
    
    # Extract features
    features = get_sentinel_features(sentinel, region, config)
    gedi_features = get_gedi_features(gedi_l4a, gedi_l2a, gedi_l2b, region, config)
    
    # Create patches
    patches, labels, patch_info = create_patches(features, gedi_features, config)
    
    # Save processed data
    output_dir = Path(config["paths"]["processed_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "patches.npy", patches)
    np.save(output_dir / "labels.npy", labels)
    patch_info.to_csv(output_dir / "patch_info.csv", index=False)
    
    logger.info(f"Created {len(patches)} patches with shape {patches.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Average AGB per patch: {np.mean(labels):.2f} Mg/ha")
    logger.info(f"Standard deviation of AGB: {np.std(labels):.2f} Mg/ha")

if __name__ == "__main__":
    main() 