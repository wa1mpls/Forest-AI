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

def download_gedi_data(config):
    """Download GEDI data using NASA CMR API"""
    logger.info("Downloading GEDI data...")
    
    gedi_dir = Path(config["paths"]["gedi_dir"])
    gedi_dir.mkdir(parents=True, exist_ok=True)
    
    # Check NASA Earthdata credentials
    username = os.environ.get('EARTHDATA_USERNAME')
    password = os.environ.get('EARTHDATA_PASSWORD')
    
    if not username or not password:
        logger.error("Please set your NASA Earthdata credentials:")
        logger.error("1. Go to https://urs.earthdata.nasa.gov/")
        logger.error("2. Create an account or sign in")
        logger.error("3. Set environment variables:")
        logger.error("   export EARTHDATA_USERNAME='your_username'")
        logger.error("   export EARTHDATA_PASSWORD='your_password'")
        return False
    
    # Create session
    session = EDLSession().auth_with_creds(username, password)
    
    # Create GeoJSON from bounds
    geojson = create_geojson_from_bounds(config["region"]["bounds"])
    poly = gpd.GeoDataFrame.from_features([geojson])
    poly.crs = 'EPSG:4326'
    
    # Set temporal range
    start_date = dt.datetime.strptime(config["date_range"]["start"], DT_FORMAT)
    end_date = dt.datetime.strptime(config["date_range"]["end"], DT_FORMAT)
    dt_cmr = '%Y-%m-%dT%H:%M:%SZ'
    temporal = start_date.strftime(dt_cmr) + ',' + end_date.strftime(dt_cmr)
    
    # Download GEDI L4A data
    doi = "10.3334/ORNLDAAC/2056"  # GEDI L4A V2.1
    for g in get_granules_names(doi, poly, temporal):
        download_files(path.join(gedi_dir, g['url'].rsplit('/', 1)[1]), session, **g)
    
    return True

def download_sentinel_data(config):
    """Download Sentinel-2 data"""
    logger.info("Downloading Sentinel-2 data...")
    
    sentinel_dir = Path(config["paths"]["sentinel_dir"])
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    
    # Sentinel-2 data is available through Google Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        logger.error("Please authenticate with Google Earth Engine first:")
        logger.error("1. Go to https://earthengine.google.com/")
        logger.error("2. Sign in with your Google account")
        logger.error("3. Run: earthengine authenticate")
        return False
    
    # Download Sentinel-2 data for the specified region and date range
    region = ee.Geometry.Rectangle([
        config["region"]["bounds"]["min_lon"],
        config["region"]["bounds"]["min_lat"],
        config["region"]["bounds"]["max_lon"],
        config["region"]["bounds"]["max_lat"]
    ])
    
    date_range = ee.DateRange(
        config["date_range"]["start"],
        config["date_range"]["end"]
    )
    
    # Get Sentinel-2 data
    sentinel = ee.ImageCollection(config["sentinel"]["collection"])
    sentinel = sentinel.filterBounds(region).filterDate(date_range)
    
    # Export to Google Drive
    task = ee.batch.Export.image.toDrive(
        image=sentinel.select(config["sentinel"]["bands"]),
        description='Sentinel_Data_Export',
        folder='Sentinel_Data',
        scale=10,
        region=region,
        fileFormat='GeoTIFF'
    )
    task.start()
    
    logger.info("Sentinel-2 data export started. Please check your Google Drive for the exported data.")
    return True

def check_dataset_availability(config):
    """Check if required datasets are available"""
    logger.info("Checking dataset availability...")
    
    # Check GEDI data
    gedi_dir = Path(config["paths"]["gedi_dir"])
    gedi_files = list(gedi_dir.glob("*.h5"))
    if not gedi_files:
        logger.warning("GEDI data not found. Will attempt to download.")
        return False
    
    # Check Sentinel-2 data
    sentinel_dir = Path(config["paths"]["sentinel_dir"])
    sentinel_files = list(sentinel_dir.glob("*.tif"))
    if not sentinel_files:
        logger.warning("Sentinel-2 data not found. Will attempt to download.")
        return False
    
    logger.info("All required datasets are available.")
    return True

def main():
    """Main function to download datasets"""
    # Load config
    config = load_config()
    
    # Check dataset availability
    if check_dataset_availability(config):
        logger.info("All datasets are already available.")
        return
    
    # Download GEDI data
    if not download_gedi_data(config):
        logger.error("Failed to download GEDI data.")
        return
    
    # Download Sentinel-2 data
    if not download_sentinel_data(config):
        logger.error("Failed to download Sentinel-2 data.")
        return
    
    logger.info("Dataset download complete!")

if __name__ == "__main__":
    main() 