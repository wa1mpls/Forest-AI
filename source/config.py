import os

# Paths
WORKING_DIR = os.path.join(os.path.expanduser("~"), "forest-ai")
DATA_DIR = os.path.join(WORKING_DIR, "data")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")

# GEDI Configuration
GEDI_L4A_COLLECTION = "LARSE/GEDI/GEDI04_A_002_MONTHLY"
GEDI_L2A_COLLECTION = "LARSE/GEDI/GEDI02_A_002_MONTHLY"
GEDI_L2B_COLLECTION = "LARSE/GEDI/GEDI02_B_002_MONTHLY"

# Sentinel-2 Configuration
SENTINEL_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_BANDS = ["B4", "B3", "B2"]  # RGB bands
SENTINEL_VIS_PARAMS = {
    "min": 0,
    "max": 3000,
    "gamma": 1.5
}

# Region of Interest
REGION = {
    "min_lon": -75.0,
    "min_lat": -15.0,
    "max_lon": -50.0,
    "max_lat": 5.0
}

# Date Range
DATE_RANGE = ("2023-01-01", "2023-12-31")

# Model Configuration
MODEL_CONFIG = {
    "num_outputs": 5,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 16,
    "num_epochs": 50,
    "patience": 5
}

# Data Configuration
DATA_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "image_size": (224, 224),
    "max_samples": 500
}

# Features Configuration
FEATURES = {
    "l4a": ['agbd', 'agbd_se', 'l2_quality_flag', 'sensitivity', 'degrade_flag', 'beam_type'],
    "l2a": ['rh100', 'rh95', 'rh75'],
    "l2b": ['cover', 'pai']
}

# Create directories if they don't exist
for dir_path in [WORKING_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True) 