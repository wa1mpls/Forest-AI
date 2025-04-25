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

def run_pipeline(mode, config_path):
    """Run the selected pipeline mode"""
    logger.info(f"Starting pipeline in {mode} mode")
    
    # Load config
    config = load_config(config_path)
    
    # Run selected mode
    if mode == "download" or mode == "all":
        logger.info("Downloading datasets...")
        download_dataset()
    
    if mode == "prepare" or mode == "all":
        logger.info("Preparing data...")
        prepare_data(config_path)
    
    if mode == "train" or mode == "all":
        logger.info("Training model...")
        train_model()
    
    if mode == "evaluate" or mode == "all":
        logger.info("Evaluating model...")
        evaluate_model()
    
    if mode == "inference" or mode == "all":
        logger.info("Running inference...")
        run_inference()
    
    logger.info("Pipeline complete!")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Run pipeline
    run_pipeline(args.mode, args.config)

if __name__ == "__main__":
    main() 