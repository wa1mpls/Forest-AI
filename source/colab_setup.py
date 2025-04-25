import os
import ee
import geemap
from pathlib import Path
import subprocess
import sys

def setup_colab():
    """Setup environment for Google Colab"""
    print("Setting up Google Colab environment...")
    
    # Install required packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "geemap"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "geopandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    
    # Create directory structure
    os.makedirs("data/raw/sentinel", exist_ok=True)
    os.makedirs("data/raw/gedi", exist_ok=True)
    os.makedirs("data/raw/shapefiles", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/outputs", exist_ok=True)
    
    # Authenticate Google Earth Engine
    try:
        # First authenticate
        ee.Authenticate()
        
        # Then initialize
        ee.Initialize()
        
        print("Successfully authenticated with Google Earth Engine!")
        return True
    except Exception as e:
        print("Error authenticating with Google Earth Engine:")
        print(e)
        print("\nPlease follow these steps:")
        print("1. Go to https://earthengine.google.com/")
        print("2. Sign in with your Google account")
        print("3. Run: earthengine authenticate")
        return False

if __name__ == "__main__":
    setup_colab() 