import ee
import geemap
import numpy as np
import rasterio
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class SentinelProcessor:
    def __init__(self, config_path):
        """Initialize Sentinel-2 processor"""
        self.config = self._load_config(config_path)
        self.ee = ee
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _calculate_spectral_indices(self, image):
        """Calculate spectral indices (NDVI, GNDVI, NBR)"""
        for idx in self.config["sentinel"]["spectral_indices"]:
            formula = idx["formula"]
            # Replace band names with actual band values
            for band in ["B2", "B3", "B4", "B8", "B11", "B12"]:
                formula = formula.replace(band, f"image.select('{band}')")
            # Calculate index
            index = ee.Image(eval(formula))
            image = image.addBands(index.rename(idx["name"]))
        return image
    
    def _apply_cloud_mask(self, image):
        """Apply cloud mask using QA60 band"""
        if self.config["sentinel"]["cloud_mask"]:
            try:
                qa = image.select('QA60')
                cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)  # Cloud mask
                shadow_mask = qa.bitwiseAnd(1 << 11).eq(0)  # Shadow mask
                return image.updateMask(cloud_mask).updateMask(shadow_mask)
            except Exception as e:
                logger.warning(f"Could not apply cloud mask: {str(e)}")
                return image
        return image
    
    def process_collection(self):
        """Process Sentinel-2 collection"""
        logger.info("Processing Sentinel-2 collection...")
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception as e:
            logger.error("Please authenticate with Google Earth Engine first:")
            logger.error("1. Go to https://earthengine.google.com/")
            logger.error("2. Sign in with your Google account")
            logger.error("3. Run: earthengine authenticate")
            return False
        
        # Define region of interest
        region = ee.Geometry.Rectangle([
            self.config["region"]["bounds"]["min_lon"],
            self.config["region"]["bounds"]["min_lat"],
            self.config["region"]["bounds"]["max_lon"],
            self.config["region"]["bounds"]["max_lat"]
        ])
        
        # Define date range
        date_range = ee.DateRange(
            self.config["date_range"]["start"],
            self.config["date_range"]["end"]
        )
        
        # Get Sentinel-2 collection
        sentinel = ee.ImageCollection(self.config["sentinel"]["collection"])
        sentinel = sentinel.filterBounds(region).filterDate(date_range)
        
        # Select bands
        sentinel = sentinel.select(self.config["sentinel"]["bands"])
        
        # Calculate spectral indices
        sentinel = sentinel.map(self._calculate_spectral_indices)
        
        # Apply cloud mask
        sentinel = sentinel.map(self._apply_cloud_mask)
        
        # Get least cloudy image
        sentinel = sentinel.sort('CLOUD_COVER').first()
        
        # Export to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=sentinel,
            description='Sentinel_Processed',
            folder='Sentinel_Data',
            scale=10,
            region=region,
            fileFormat='GeoTIFF'
        )
        task.start()
        
        logger.info("Sentinel-2 processing complete. Please check your Google Drive for the exported data.")
        return True
    
    def create_patches(self, image_path):
        """Create patches from Sentinel-2 image"""
        logger.info("Creating patches from Sentinel-2 image...")
        
        # Read image
        with rasterio.open(image_path) as src:
            image = src.read()
            transform = src.transform
        
        # Get patch size from config
        patch_size = self.config["sentinel"]["patch_size"]
        
        # Check image dimensions
        if image.shape[1] < patch_size or image.shape[2] < patch_size:
            logger.error(f"Image dimensions {image.shape[1:]} are smaller than patch size {patch_size}")
            return False
        
        # Create patches
        patches = []
        for i in range(0, image.shape[1] - patch_size + 1, patch_size):
            for j in range(0, image.shape[2] - patch_size + 1, patch_size):
                patch = image[:, i:i+patch_size, j:j+patch_size]
                
                # Check patch dimensions
                if patch.shape[1:] != (patch_size, patch_size):
                    logger.warning(f"Skipping incomplete patch at ({i}, {j})")
                    continue
                
                patches.append(patch)
        
        # Save patches
        patches_dir = Path(self.config["paths"]["patches_dir"])
        patches_dir.mkdir(parents=True, exist_ok=True)
        
        for i, patch in enumerate(patches):
            patch_path = patches_dir / f"patch_{i}.npy"
            np.save(patch_path, patch)
        
        logger.info(f"Created {len(patches)} patches")
        return True 