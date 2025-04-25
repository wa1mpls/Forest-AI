import os
import yaml
import requests
import zipfile
import gdown
from pathlib import Path
from tqdm import tqdm
import ee
import geemap
import argparse
import http.cookiejar
import hashlib
import sys
import datetime as dt
import geopandas as gpd
from os import path
from shapely.ops import orient
from urllib.parse import urlsplit
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

# CMR API base url
CMR_URL = "https://cmr.earthdata.nasa.gov/search/"
AUTH_HOST = "https://urs.earthdata.nasa.gov"
EDL_AUTH = "https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget"
DT_FORMAT = "%Y-%m-%d"
GRANULE_FORMAT = "h5"

class EDLSession(requests.Session):
    """Creates a NASA EarthData Login session"""
    def __init__(self):
        super().__init__()

    def auth_with_creds(self, username: str, password: str):
        self.auth = (username, password)
        self.get(AUTH_HOST)
        if "urs_user_already_logged" not in self.cookies.get_dict():
            raise Exception("Username or password is incorrect")
        return self

    def auth_with_token(self, token: str):
        self.headers.update({'Authorization': 'Bearer {0}'.format(token)})
        return self

    def auth_with_cookiejar(self, cookies: http.cookiejar):
        self.cookies = cookies
        return self

def check_sha256(granule_url: str, local_file: str):
    """Checks if the sha256 hashes of local and remote file are same"""
    response = requests.get(granule_url)
    response.raise_for_status()
    sha256_1 = response.content.decode("utf-8")
    sha256_2 = hashlib.sha256()
    with open(local_file,'rb') as f:
        while True: 
            data = f.read(4096)
            if len(data) == 0:
                break
            else:
                sha256_2.update(data)
    return sha256_1 == sha256_2.hexdigest()

def download_files(local_file: str, session, **granule):
    """Downloads the granules"""
    if session is None:
        session = EDLSession()
    
    if path.isfile(local_file) and granule['sha256'] and check_sha256(granule['sha256'], local_file):
        logger.info(f'{path.basename(local_file)} is already downloaded at {path.dirname(local_file)}')
    else:
        logger.info(f'Downloading {path.basename(local_file)} ...')
        try:
            with session.get(granule['url'], stream=True) as r:
                r.raise_for_status()
                with open(local_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        except requests.exceptions.HTTPError as e:
            raise Exception(f"{e.response}.\r\n Set up NASA Earthdata Login authentication at {EDL_AUTH}")

def get_granules_names(doi: str, poly_epsg4326, temporal_str: str):
    """Get url and sha256 of granules that overlaps the temporal and spatial bounds"""
    logger.info("Searching for granules ..")

    # orienting coordinates clockwise
    poly_epsg4326.geometry = poly_epsg4326.geometry.apply(orient, args=(1,))

    # reducing number of vertices in the polygon
    grsm_epsg4326 = poly_epsg4326.simplify(0.0005)

    doisearch = requests.get(CMR_URL + 'collections.json?doi=' + doi).json()['feed']['entry'][0]
    concept_id = doisearch['id']
    data_center = doisearch['data_center']
    geojson = {"shapefile": ("poly.json", poly_epsg4326.geometry.to_json(), "application/geo+json")}

    page_num = 1
    page_size = 2000 # CMR page size limit

    granule_arr = []

    while True:
        # defining parameters
        cmr_param = {
            "collection_concept_id": concept_id, 
            "page_size": page_size,
            "page_num": page_num,
            "temporal": temporal_str,
            "simplify-shapefile": 'true'
        }
        
        granulesearch = CMR_URL + 'granules.json'
        response = requests.post(granulesearch, data=cmr_param, files=geojson)
        granules = response.json()['feed']['entry']
        
        if granules:
            for g in granules:          
                # Get URL of HDF5 files
                href=''
                sha256 =''
                for links in g['links']:
                    if 'href' in links:
                        if not data_center.startswith('ORNL'): 
                            if links['href'].endswith(GRANULE_FORMAT):
                                href = links['href']
                        else:
                            if links['href'].endswith(GRANULE_FORMAT) and links['title'].startswith('Download'):
                                href = links['href']
                            if links['href'].endswith('.sha256'):
                                sha256 = links['href']
                                
                granule_arr.append({'url':href, 'sha256':sha256})
            page_num += 1   
        else: 
            break
        
        logger.info(f"Total granules found: {len(granule_arr)}")
    return granule_arr

def create_geojson_from_bounds(bounds):
    """Create GeoJSON polygon from bounds"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bounds['min_lon'], bounds['min_lat']],
                [bounds['max_lon'], bounds['min_lat']],
                [bounds['max_lon'], bounds['max_lat']],
                [bounds['min_lon'], bounds['max_lat']],
                [bounds['min_lon'], bounds['min_lat']]
            ]]
        },
        "properties": {}
    }

def download_amazon_shapefile(config):
    """Download Amazon forest shapefile"""
    logger.info("Downloading Amazon forest shapefile...")
    
    shapefile_dir = Path(config["paths"]["shapefile_dir"])
    shapefile_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from Global Forest Watch
    url = "https://data.globalforestwatch.org/datasets/amazon-forest-cover-2000/geoservice"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save shapefile
        with open(shapefile_dir / "amazon_forest.shp", "wb") as f:
            f.write(response.content)
        logger.info("Amazon forest shapefile downloaded successfully")
        return True
    else:
        logger.error("Failed to download Amazon forest shapefile")
        return False

def get_sentinel_data(config):
    """Get Sentinel-2 data from Google Earth Engine"""
    logger.info("Getting Sentinel-2 data from Google Earth Engine...")
    
    # Initialize Earth Engine
    try:
        ee.Initialize(project='ee-ngonguyenthanhthanh00')
    except Exception as e:
        logger.error("Please authenticate with Google Earth Engine first:")
        logger.error("1. Go to https://earthengine.google.com/")
        logger.error("2. Sign in with your Google account")
        logger.error("3. Run: earthengine authenticate")
        return None, None
    
    # Define region of interest
    region = ee.Geometry.Rectangle([
        config["region"]["bounds"]["min_lon"],
        config["region"]["bounds"]["min_lat"],
        config["region"]["bounds"]["max_lon"],
        config["region"]["bounds"]["max_lat"]
    ])
    
    # Define date range
    start_date = ee.Date(config["date_range"]["start"])
    end_date = ee.Date(config["date_range"]["end"])
    
    # Get Sentinel-2 collection
    sentinel = ee.ImageCollection(config["sentinel"]["collection"])
    
    # Filter by date and region
    sentinel = sentinel.filterBounds(region).filterDate(start_date, end_date)
    
    # Get least cloudy image
    sentinel = sentinel.sort('CLOUD_COVERAGE_ASSESSMENT').first()
    
    # Select bands and QA60
    bands = config["sentinel"]["bands"] + ["QA60"]
    sentinel = sentinel.select(bands)
    
    return sentinel, region

def get_gedi_data(config):
    """Get GEDI data from Google Earth Engine"""
    logger.info("Getting GEDI data from Google Earth Engine...")
    
    # Define region of interest
    region = ee.Geometry.Rectangle([
        config["region"]["bounds"]["min_lon"],
        config["region"]["bounds"]["min_lat"],
        config["region"]["bounds"]["max_lon"],
        config["region"]["bounds"]["max_lat"]
    ])
    
    # Define date range
    start_date = ee.Date(config["date_range"]["start"])
    end_date = ee.Date(config["date_range"]["end"])
    
    # Get GEDI collections
    gedi_l4a = ee.ImageCollection(config["gedi"]["collections"]["l4a"])
    gedi_l2a = ee.ImageCollection(config["gedi"]["collections"]["l2a"])
    gedi_l2b = ee.ImageCollection(config["gedi"]["collections"]["l2b"])
    
    # Filter by date and region
    gedi_l4a = gedi_l4a.filterBounds(region).filterDate(start_date, end_date)
    gedi_l2a = gedi_l2a.filterBounds(region).filterDate(start_date, end_date)
    gedi_l2b = gedi_l2b.filterBounds(region).filterDate(start_date, end_date)
    
    # Select features
    gedi_l4a = gedi_l4a.select(config["gedi"]["features"]["l4a"])
    gedi_l2a = gedi_l2a.select(config["gedi"]["features"]["l2a"])
    gedi_l2b = gedi_l2b.select(config["gedi"]["features"]["l2b"])
    
    return gedi_l4a, gedi_l2a, gedi_l2b, region

def main():
    """Main function to get datasets from Google Earth Engine"""
    # Load config
    config = load_config()
    
    # Download Amazon shapefile
    if not download_amazon_shapefile(config):
        logger.error("Failed to download Amazon shapefile")
        return
    
    # Get Sentinel-2 data
    sentinel, region = get_sentinel_data(config)
    if sentinel is None or region is None:
        logger.error("Failed to get Sentinel-2 data")
        return
    
    # Get GEDI data
    gedi_l4a, gedi_l2a, gedi_l2b, region = get_gedi_data(config)
    if not gedi_l4a or not gedi_l2a or not gedi_l2b:
        logger.error("Failed to get GEDI data")
        return
    
    logger.info("Successfully retrieved all datasets from Google Earth Engine!")

if __name__ == "__main__":
    main() 