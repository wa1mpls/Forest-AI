import os
import argparse
import yaml
from pathlib import Path
import sys

from data.download_dataset import main as download_dataset
from data.preprocessing.sentinel_preprocessing import SentinelProcessor
from data.preprocessing.gedi_preprocessing import GEDIProcessor
from train import main as train_model
from evaluate import main as evaluate_model
from inference import main as run_inference
from utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Forest AI Pipeline")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["download", "prepare", "train", "evaluate", "inference", "all"],
        default="all",
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to config file"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        sys.exit(1)

def prepare_data(config_path):
    """Prepare data for training"""
    logger.info("Preparing data...")
    
    # Process Sentinel-2 data
    sentinel = SentinelProcessor(config_path)
    if not sentinel.process_collection():
        logger.error("Failed to process Sentinel-2 data")
        return False
    
    # Process GEDI data
    gedi = GEDIProcessor(config_path)
    gedi_files = list(Path(config["paths"]["gedi_dir"]).glob("*.h5"))
    if not gedi.process_gedi_files(gedi_files):
        logger.error("Failed to process GEDI data")
        return False
    
    # Create training dataset
    if not gedi.create_training_dataset(config["paths"]["patches_dir"]):
        logger.error("Failed to create training dataset")
        return False
    
    logger.info("Data preparation complete!")
    return True

def run_pipeline(mode="all", config_path=None):
    """Run the complete pipeline"""
    logger.info(f"Starting pipeline in {mode} mode")
    
    if mode == "all" or mode == "download":
        logger.info("Downloading datasets...")
        from data.download_dataset import main as download_main
        download_main()
    
    if mode == "all" or mode == "preprocess":
        logger.info("Preparing data...")
        from data.preprocess_data import main as preprocess_main
        preprocess_main()
    
    if mode == "all" or mode == "train":
        logger.info("Training model...")
        from train import main as train_main
        train_main()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Forest AI Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                      choices=["all", "download", "preprocess", "train"],
                      help="Pipeline mode to run")
    parser.add_argument("--config", type=str, default=None,
                      help="Path to config file")
    args = parser.parse_args()
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        # Setup Colab environment
        from colab_setup import setup_colab
        if not setup_colab():
            logger.error("Failed to setup Colab environment")
            return
    
    # Run pipeline
    run_pipeline(args.mode, args.config)

if __name__ == "__main__":
    main() 