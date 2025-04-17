import ee
import os
import json
import requests
from PIL import Image
from tqdm import tqdm
from ..config import *

def get_gedi_l4a_data(point, date_range):
    """Get GEDI L4A data for a point"""
    gedi_l4a = ee.ImageCollection(GEDI_L4A_COLLECTION)
    filtered = gedi_l4a.filterBounds(point).filterDate(date_range[0], date_range[1])
    
    properties = {}
    for feat in FEATURES['l4a']:
        if feat in ['l2_quality_flag', 'degrade_flag', 'beam_type']:
            properties[feat] = filtered.select(feat).mode()
        else:
            properties[feat] = filtered.select(feat).mean()
    
    return properties

def get_gedi_l2a_data(point, date_range):
    """Get GEDI L2A data for a point"""
    gedi_l2a = ee.ImageCollection(GEDI_L2A_COLLECTION)
    filtered = gedi_l2a.filterBounds(point).filterDate(date_range[0], date_range[1])
    
    properties = {}
    for feat in FEATURES['l2a']:
        properties[feat] = filtered.select(feat).mean()
    
    return properties

def get_gedi_l2b_data(point, date_range):
    """Get GEDI L2B data for a point"""
    gedi_l2b = ee.ImageCollection(GEDI_L2B_COLLECTION)
    filtered = gedi_l2b.filterBounds(point).filterDate(date_range[0], date_range[1])
    
    properties = {}
    for feat in FEATURES['l2b']:
        properties[feat] = filtered.select(feat).mean()
    
    return properties

def get_sentinel_image(lat, lon, output_path, date_range):
    """Get Sentinel-2 image and GEDI data for a point"""
    point = ee.Geometry.Point([lon, lat])
    
    # Get Sentinel-2 image
    image = (
        ee.ImageCollection(SENTINEL_COLLECTION)
        .filterBounds(point)
        .filterDate(date_range[0], date_range[1])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .median()
        .clip(point.buffer(2500))
    )
    
    # Get GEDI data
    gedi_l4a = get_gedi_l4a_data(point, date_range)
    gedi_l2a = get_gedi_l2a_data(point, date_range)
    gedi_l2b = get_gedi_l2b_data(point, date_range)
    
    # Combine all data
    all_data = {**gedi_l4a, **gedi_l2a, **gedi_l2b}
    
    try:
        # Get Sentinel-2 image
        url = image.getThumbURL({
            "dimensions": "256x256",
            "region": point.buffer(2500),
            "bands": SENTINEL_BANDS,
            **SENTINEL_VIS_PARAMS
        })
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Save image
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # Save metadata
            metadata_path = output_path.replace('.png', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(all_data, f)
            
            print(f"✅ Saved {output_path} and metadata")
        else:
            print(f"❌ Failed to download image from {url}")
            
    except Exception as e:
        print(f"❌ Error getting data at ({lat}, {lon}): {e}")

def download_data(df, output_dir, date_range, max_samples=None):
    """Download data for all points in dataframe"""
    if max_samples is None:
        max_samples = DATA_CONFIG['max_samples']
    
    df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        lat, lon = row["lat_lowestmode_a1"], row["lon_lowestmode_a1"]
        output_path = os.path.join(output_dir, f"image_{i}.png")
        get_sentinel_image(lat, lon, output_path, date_range)
    
    print(f"✅ Completed! Images saved from image_0.png to image_{len(df)-1}.png") 