#!/bin/bash

# Create directory structure
mkdir -p data/raw/sentinel
mkdir -p data/raw/gedi
mkdir -p data/raw/shapefiles
mkdir -p data/processed
mkdir -p data/outputs

# Install required packages
pip install earthengine-api
pip install rasterio
pip install geopandas
pip install h5py
pip install numpy
pip install pandas
pip install tqdm
pip install pyyaml

# Check NASA Earthdata credentials
if [ -z "$EARTHDATA_USERNAME" ] || [ -z "$EARTHDATA_PASSWORD" ]; then
    echo "Please set your NASA Earthdata credentials:"
    echo "1. Go to https://urs.earthdata.nasa.gov/"
    echo "2. Create an account or sign in"
    echo "3. Set environment variables:"
    echo "   export EARTHDATA_USERNAME='your_username'"
    echo "   export EARTHDATA_PASSWORD='your_password'"
    exit 1
fi

# Authenticate with Google Earth Engine
echo "Authenticating with Google Earth Engine..."
earthengine authenticate

# Run the pipeline
echo "Running the pipeline..."

# 1. Download datasets
python source/data/download_dataset.py

# 2. Preprocess data
python source/data/preprocess_data.py

# 3. Train model
python source/train.py

echo "Setup complete!" 