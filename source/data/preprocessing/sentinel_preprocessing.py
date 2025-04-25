import ee
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
from pathlib import Path
import yaml
from typing import Tuple, Dict, List
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentinelProcessor:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize Sentinel-2 processor
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception as e:
            logger.error("Failed to initialize Earth Engine. Please authenticate first.")
            raise e
        
        # Set up paths
        self.sentinel_dir = Path(self.config['paths']['sentinel_dir'])
        self.sentinel_dir.mkdir(parents=True, exist_ok=True)
        
    def get_sentinel_collection(self) -> ee.ImageCollection:
        """
        Get Sentinel-2 image collection for the specified region and date range
        
        Returns:
            ee.ImageCollection: Filtered Sentinel-2 collection
        """
        # Get region and date range from config
        region = ee.Geometry.Rectangle([
            self.config['region']['bounds']['min_lon'],
            self.config['region']['bounds']['min_lat'],
            self.config['region']['bounds']['max_lon'],
            self.config['region']['bounds']['max_lat']
        ])
        
        date_range = (
            self.config['date_range']['start'],
            self.config['date_range']['end']
        )
        
        # Get Sentinel-2 collection
        collection = ee.ImageCollection(self.config['sentinel']['collection'])\
            .filterBounds(region)\
            .filterDate(*date_range)\
            .select(self.config['sentinel']['bands'])
        
        return collection
    
    def apply_cloud_mask(self, image: ee.Image) -> ee.Image:
        """
        Apply cloud mask to Sentinel-2 image
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            ee.Image: Masked image
        """
        # Get QA60 band for cloud mask
        qa = image.select('QA60')
        
        # Create cloud mask
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
        cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
        
        # Apply masks
        masked_image = image.updateMask(cloud_mask).updateMask(cirrus_mask)
        
        return masked_image
    
    def calculate_spectral_indices(self, image: ee.Image) -> ee.Image:
        """
        Calculate spectral indices for Sentinel-2 image
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            ee.Image: Image with added spectral indices
        """
        for index in self.config['sentinel']['spectral_indices']:
            formula = index['formula']
            name = index['name']
            
            # Replace band names with actual band values
            for band in self.config['sentinel']['bands']:
                formula = formula.replace(band, f'image.select("{band}")')
            
            # Calculate index
            index_value = eval(formula)
            image = image.addBands(index_value.rename(name))
        
        return image
    
    def download_image(self, image: ee.Image, filename: str) -> None:
        """
        Download Sentinel-2 image to local storage
        
        Args:
            image: Sentinel-2 image
            filename: Output filename
        """
        # Get region
        region = ee.Geometry.Rectangle([
            self.config['region']['bounds']['min_lon'],
            self.config['region']['bounds']['min_lat'],
            self.config['region']['bounds']['max_lon'],
            self.config['region']['bounds']['max_lat']
        ])
        
        # Get download URL
        url = image.getDownloadURL({
            'scale': 10,  # 10m resolution
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        # Download image
        output_path = self.sentinel_dir / filename
        os.system(f'curl -o {output_path} "{url}"')
        
    def process_collection(self) -> None:
        """
        Process entire Sentinel-2 collection
        """
        # Get collection
        collection = self.get_sentinel_collection()
        
        # Get least cloudy image
        image = collection.sort('CLOUD_COVER').first()
        
        # Apply cloud mask if enabled
        if self.config['sentinel']['cloud_mask']:
            image = self.apply_cloud_mask(image)
        
        # Calculate spectral indices
        image = self.calculate_spectral_indices(image)
        
        # Download image
        self.download_image(image, 'sentinel_image.tif')
        
        logger.info("Sentinel-2 processing completed successfully")
    
    def create_patches(self, image_path: str, patch_size: int = None) -> None:
        """
        Create patches from Sentinel-2 image
        
        Args:
            image_path: Path to Sentinel-2 image
            patch_size: Size of patches (default from config)
        """
        if patch_size is None:
            patch_size = self.config['sentinel']['patch_size']
        
        # Read image
        with rasterio.open(image_path) as src:
            image = src.read()
            transform = src.transform
            
        # Get image dimensions
        height, width = image.shape[1:]
        
        # Create patches
        patches = []
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                # Get patch
                patch = image[:, i:i+patch_size, j:j+patch_size]
                
                # Skip if patch is incomplete
                if patch.shape[1:] != (patch_size, patch_size):
                    continue
                
                # Save patch
                patch_path = self.sentinel_dir / 'patches' / f'patch_{i}_{j}.npy'
                np.save(patch_path, patch)
                
                patches.append({
                    'path': str(patch_path),
                    'coordinates': (i, j),
                    'transform': transform * rasterio.Affine.translation(j, i)
                })
        
        # Save patch metadata
        metadata_path = self.sentinel_dir / 'patches' / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(patches, f)
        
        logger.info(f"Created {len(patches)} patches from Sentinel-2 image") 