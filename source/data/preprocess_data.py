import os
import yaml
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from utils.logger import get_logger
import h5py
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import Point, box

logger = get_logger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_sentinel_data(config):
    """Load and preprocess Sentinel-2 data"""
    logger.info("Loading Sentinel-2 data...")
    
    sentinel_dir = Path(config["paths"]["sentinel_dir"])
    shapefile_path = Path(config["paths"]["shapefile_dir"]) / "amazon_forest.shp"
    
    # Load shapefile
    amazon_forest = gpd.read_file(shapefile_path)
    
    # Load Sentinel-2 image
    with rasterio.open(sentinel_dir / "sentinel_image.tif") as src:
        # Clip to Amazon forest
        image, transform = rasterio.mask.mask(src, amazon_forest.geometry, crop=True)
        
        # Apply cloud mask using QA60 band
        cloud_mask = image[config["sentinel"]["cloud_mask_band"]]
        image = np.where(cloud_mask > 0, 0, image)
        
        # Calculate spectral indices
        ndvi = (image[config["sentinel"]["bands"].index("B8")] - 
                image[config["sentinel"]["bands"].index("B4")]) / \
               (image[config["sentinel"]["bands"].index("B8")] + 
                image[config["sentinel"]["bands"].index("B4")])
        
        gndvi = (image[config["sentinel"]["bands"].index("B8")] - 
                 image[config["sentinel"]["bands"].index("B3")]) / \
                (image[config["sentinel"]["bands"].index("B8")] + 
                 image[config["sentinel"]["bands"].index("B3")])
        
        nbr = (image[config["sentinel"]["bands"].index("B8")] - 
               image[config["sentinel"]["bands"].index("B12")]) / \
              (image[config["sentinel"]["bands"].index("B8")] + 
               image[config["sentinel"]["bands"].index("B12")])
        
        # Stack all features
        features = np.stack([
            image[config["sentinel"]["bands"].index(b)] for b in config["sentinel"]["bands"]
        ] + [ndvi, gndvi, nbr])
        
        return features, transform

def load_gedi_data(config):
    """Load and preprocess GEDI data"""
    logger.info("Loading GEDI data...")
    
    gedi_dir = Path(config["paths"]["gedi_dir"])
    gedi_files = list(gedi_dir.glob("*.h5"))
    
    gedi_data = []
    for file in gedi_files:
        with h5py.File(file, 'r') as f:
            # Filter data based on quality criteria
            mask = (
                (f['l2_quality_flag'][:] == config["gedi"]["filters"]["quality_flag"]) &
                (f['rh100'][:] > config["gedi"]["filters"]["min_rh100"]) &
                (f['slope'][:] < config["gedi"]["filters"]["max_slope"]) &
                (f['confidence'][:] > config["gedi"]["filters"]["confidence"])
            )
            
            gedi_data.append({
                'lat': f['lat'][:][mask],
                'lon': f['lon'][:][mask],
                'agbd': f['agbd'][:][mask]
            })
    
    return pd.concat([pd.DataFrame(d) for d in gedi_data])

def normalize_coordinates(gedi_data, sentinel_transform):
    """Normalize GEDI coordinates to Sentinel-2 image coordinates"""
    logger.info("Normalizing coordinates...")
    
    # Convert GEDI coordinates to Sentinel-2 image coordinates
    gedi_points = []
    for lat, lon in zip(gedi_data['lat'], gedi_data['lon']):
        x, y = rasterio.transform.rowcol(sentinel_transform, lon, lat)
        gedi_points.append({'x': x, 'y': y, 'lat': lat, 'lon': lon})
    
    return pd.DataFrame(gedi_points)

def check_data_quality(features, gedi_data, config):
    """Check quality of input data"""
    logger.info("Checking data quality...")
    
    # Check Sentinel-2 data
    if features.shape[0] != config["preprocessing"]["num_features"]:
        logger.error(f"Expected {config['preprocessing']['num_features']} features, got {features.shape[0]}")
        return False
    
    # Check GEDI data
    if len(gedi_data) < config["preprocessing"]["min_gedi_points"]:
        logger.error(f"Not enough GEDI points (minimum: {config['preprocessing']['min_gedi_points']})")
        return False
    
    # Check for NaN values
    if np.isnan(features).any():
        logger.error("NaN values found in features")
        return False
    
    if gedi_data['agbd'].isna().any():
        logger.error("NaN values found in GEDI data")
        return False
    
    return True

def create_patches(features, transform, gedi_data, config):
    """Create patches and assign GEDI labels"""
    logger.info("Creating patches...")
    
    patch_size = config["preprocessing"]["image_size"][0]
    num_features = config["preprocessing"]["num_features"]
    
    # Normalize coordinates
    gedi_coords = normalize_coordinates(gedi_data, transform)
    
    # Calculate number of patches
    height, width = features.shape[1:]
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    patches = []
    labels = []
    patch_info = []
    
    for i in tqdm(range(num_patches_h)):
        for j in range(num_patches_w):
            # Extract patch
            patch = features[:, 
                           i*patch_size:(i+1)*patch_size,
                           j*patch_size:(j+1)*patch_size]
            
            # Create patch bounding box
            patch_box = box(j*patch_size, i*patch_size, 
                          (j+1)*patch_size, (i+1)*patch_size)
            
            # Find GEDI points in this patch
            mask = (
                (gedi_coords['x'] >= j*patch_size) & 
                (gedi_coords['x'] < (j+1)*patch_size) &
                (gedi_coords['y'] >= i*patch_size) & 
                (gedi_coords['y'] < (i+1)*patch_size)
            )
            
            if mask.sum() >= config["preprocessing"]["min_gedi_points"]:
                patches.append(patch)
                labels.append(gedi_data.loc[mask, 'agbd'].mean())
                
                # Save patch information
                patch_info.append({
                    'patch_id': len(patches) - 1,
                    'x': j*patch_size,
                    'y': i*patch_size,
                    'num_gedi_points': mask.sum(),
                    'mean_agbd': labels[-1],
                    'std_agbd': gedi_data.loc[mask, 'agbd'].std()
                })
    
    return np.array(patches), np.array(labels), pd.DataFrame(patch_info)

def main():
    """Main preprocessing function"""
    # Load config
    config = load_config()
    
    # Load and preprocess data
    features, transform = load_sentinel_data(config)
    gedi_data = load_gedi_data(config)
    
    # Check data quality
    if not check_data_quality(features, gedi_data, config):
        logger.error("Data quality check failed")
        return
    
    # Create patches
    patches, labels, patch_info = create_patches(features, transform, gedi_data, config)
    
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