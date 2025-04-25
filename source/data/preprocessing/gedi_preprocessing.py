import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin
import os
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import logging
import json
from utils.logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GEDIProcessor:
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize GEDI processor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Set up paths
        self.gedi_dir = Path(self.config['paths']['gedi_dir'])
        self.gedi_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def read_gedi_file(self, file_path: str) -> pd.DataFrame:
        """
        Read GEDI L4A HDF5 file
        
        Args:
            file_path: Path to GEDI HDF5 file
            
        Returns:
            pd.DataFrame: GEDI data
        """
        with h5py.File(file_path, 'r') as f:
            # Get beam data
            beams = [k for k in f.keys() if k.startswith('BEAM')]
            
            # Initialize lists to store data
            data = []
            
            for beam in beams:
                # Get beam group
                beam_group = f[beam]
                
                # Get required fields
                lat = beam_group['lat_lowestmode'][:]
                lon = beam_group['lon_lowestmode'][:]
                agbd = beam_group['agbd'][:]
                agbd_se = beam_group['agbd_se'][:]
                quality = beam_group['l2_quality_flag'][:]
                sensitivity = beam_group['sensitivity'][:]
                degrade = beam_group['degrade_flag'][:]
                rh100 = beam_group['rh100'][:]
                
                # Create DataFrame for this beam
                beam_data = pd.DataFrame({
                    'lat': lat,
                    'lon': lon,
                    'agbd': agbd,
                    'agbd_se': agbd_se,
                    'quality': quality,
                    'sensitivity': sensitivity,
                    'degrade': degrade,
                    'rh100': rh100,
                    'beam': beam
                })
                
                data.append(beam_data)
            
            # Combine all beams
            df = pd.concat(data, ignore_index=True)
            
            return df
    
    def filter_gedi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter GEDI data based on quality criteria
        
        Args:
            df: GEDI DataFrame
            
        Returns:
            pd.DataFrame: Filtered GEDI data
        """
        # Apply filters from config
        filters = self.config['gedi']['filters']
        
        filtered_df = df[
            (df['quality'] == filters['quality_flag']) &
            (df['sensitivity'] > filters['confidence']) &
            (df['rh100'] > filters['min_rh100']) &
            (df['degrade'] == 0)
        ]
        
        return filtered_df
    
    def create_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Convert GEDI DataFrame to GeoDataFrame
        
        Args:
            df: GEDI DataFrame
            
        Returns:
            gpd.GeoDataFrame: GEDI data with geometry
        """
        # Create geometry from lat/lon
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def rasterize_gedi_data(self, gdf: gpd.GeoDataFrame, output_path: str) -> None:
        """
        Rasterize GEDI data to GeoTIFF
        
        Args:
            gdf: GEDI GeoDataFrame
            output_path: Output GeoTIFF path
        """
        # Get region bounds
        bounds = self.config['region']['bounds']
        
        # Create transform
        transform = from_origin(
            bounds['min_lon'],
            bounds['max_lat'],
            0.00025,  # ~30m resolution
            0.00025
        )
        
        # Calculate raster dimensions
        width = int((bounds['max_lon'] - bounds['min_lon']) / 0.00025)
        height = int((bounds['max_lat'] - bounds['min_lat']) / 0.00025)
        
        # Create raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            # Rasterize AGBD values
            shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.agbd))
            burned = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0
            )
            
            dst.write(burned, 1)
    
    def process_gedi_files(self, file_paths: List[str]) -> None:
        """
        Process multiple GEDI files
        
        Args:
            file_paths: List of GEDI file paths
        """
        # Initialize list to store all data
        all_data = []
        
        # Process each file
        for file_path in file_paths:
            logger.info(f"Processing {file_path}")
            
            # Read file
            df = self.read_gedi_file(file_path)
            
            # Filter data
            filtered_df = self.filter_gedi_data(df)
            
            all_data.append(filtered_df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create GeoDataFrame
        gdf = self.create_geodataframe(combined_df)
        
        # Save to CSV
        csv_path = self.gedi_dir / 'gedi_data.csv'
        gdf.to_csv(csv_path, index=False)
        
        # Rasterize data
        tif_path = self.gedi_dir / 'gedi_agbd.tif'
        self.rasterize_gedi_data(gdf, str(tif_path))
        
        logger.info(f"Processed {len(file_paths)} GEDI files")
        logger.info(f"Total valid points: {len(gdf)}")
        
    def match_with_sentinel(self, gedi_path: str, sentinel_path: str) -> pd.DataFrame:
        """
        Match GEDI points with Sentinel-2 patches
        
        Args:
            gedi_path: Path to GEDI CSV file
            sentinel_path: Path to Sentinel-2 patches metadata
            
        Returns:
            pd.DataFrame: Matched data
        """
        # Read GEDI data
        gedi_df = pd.read_csv(gedi_path)
        gedi_gdf = gpd.GeoDataFrame(
            gedi_df,
            geometry=gpd.points_from_xy(gedi_df.lon, gedi_df.lat),
            crs='EPSG:4326'
        )
        
        # Read Sentinel patches metadata
        with open(sentinel_path, 'r') as f:
            patches = json.load(f)
        
        # Create DataFrame for patches
        patches_df = pd.DataFrame(patches)
        patches_gdf = gpd.GeoDataFrame(
            patches_df,
            geometry=gpd.points_from_xy(
                patches_df.coordinates.apply(lambda x: x[1]),
                patches_df.coordinates.apply(lambda x: x[0])
            ),
            crs='EPSG:4326'
        )
        
        # Spatial join
        matched = gpd.sjoin(gedi_gdf, patches_gdf, how='inner', op='within')
        
        # Group by patch and calculate mean AGBD
        result = matched.groupby('path').agg({
            'agbd': 'mean',
            'agbd_se': 'mean',
            'rh100': 'mean'
        }).reset_index()
        
        return result

    def _filter_points(self, df):
        """Filter GEDI points based on quality criteria"""
        filters = self.config["gedi"]["filters"]
        
        # Apply filters
        mask = (
            (df["confidence"] > filters["confidence"]) &
            (df["quality_flag"] == filters["quality_flag"]) &
            (df["rh100"] > filters["min_rh100"]) &
            (df["slope"] < filters["max_slope"])
        )
        
        return df[mask]
    
    def _extract_features(self, h5_file, features):
        """Extract features from GEDI H5 file"""
        data = {}
        with h5py.File(h5_file, "r") as f:
            for feature in features:
                if feature in f:
                    data[feature] = f[feature][:]
        return data
    
    def process_gedi_files(self, h5_files):
        """Process GEDI H5 files"""
        logger.info("Processing GEDI files...")
        
        # Create output directory
        gedi_dir = Path(self.config["paths"]["gedi_dir"])
        gedi_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        all_data = []
        for h5_file in h5_files:
            try:
                # Extract L4A features
                l4a_data = self._extract_features(
                    h5_file,
                    self.config["gedi"]["features"]["l4a"]
                )
                
                # Extract L2A features
                l2a_data = self._extract_features(
                    h5_file,
                    self.config["gedi"]["features"]["l2a"]
                )
                
                # Extract L2B features
                l2b_data = self._extract_features(
                    h5_file,
                    self.config["gedi"]["features"]["l2b"]
                )
                
                # Combine data
                data = {**l4a_data, **l2a_data, **l2b_data}
                all_data.append(pd.DataFrame(data))
                
            except Exception as e:
                logger.error(f"Error processing {h5_file}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("No valid data found in GEDI files")
            return False
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter points
        df = self._filter_points(df)
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326"
        )
        
        # Save to CSV
        output_path = gedi_dir / "gedi_points.csv"
        gdf.to_csv(output_path, index=False)
        
        logger.info(f"Processed {len(gdf)} GEDI points")
        return True
    
    def create_training_dataset(self, sentinel_patches_dir):
        """Create training dataset by matching GEDI points with Sentinel patches"""
        logger.info("Creating training dataset...")
        
        # Load GEDI points
        gedi_dir = Path(self.config["paths"]["gedi_dir"])
        gdf = gpd.read_file(gedi_dir / "gedi_points.csv")
        
        # Load Sentinel patches
        patches_dir = Path(sentinel_patches_dir)
        patch_files = list(patches_dir.glob("*.npy"))
        
        # Create training data
        X_train = []
        y_train = []
        
        for patch_file in patch_files:
            # Load patch
            patch = np.load(patch_file)
            
            # Get patch coordinates from filename
            i, j = map(int, patch_file.stem.split("_")[1:])
            
            # Find GEDI points in this patch
            patch_bounds = [
                (i * self.config["sentinel"]["patch_size"], j * self.config["sentinel"]["patch_size"]),
                ((i + 1) * self.config["sentinel"]["patch_size"], (j + 1) * self.config["sentinel"]["patch_size"])
            ]
            
            points_in_patch = gdf[
                (gdf["lon"] >= patch_bounds[0][0]) &
                (gdf["lon"] < patch_bounds[1][0]) &
                (gdf["lat"] >= patch_bounds[0][1]) &
                (gdf["lat"] < patch_bounds[1][1])
            ]
            
            if len(points_in_patch) > 0:
                # Calculate average AGB for this patch
                avg_agb = points_in_patch["agbd"].mean()
                
                # Add to training data
                X_train.append(patch)
                y_train.append(avg_agb)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Save training data
        output_dir = Path(self.config["paths"]["processed_data"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "y_train.npy", y_train)
        
        logger.info(f"Created training dataset with {len(X_train)} samples")
        return True 