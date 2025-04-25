# source/prepare_dataset.py
import os
import yaml
import pandas as pd
from pathlib import Path
from data.preprocessing.gedi_preprocessing import GEDIProcessor
from data.preprocessing.sentinel_preprocessing import SentinelProcessor

def load_config():
    config_dir = Path(__file__).resolve().parent / "configs"
    with open(config_dir / "data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    return data_config

def main():
    config = load_config()

    # 1. Xử lý GEDI
    gedi = GEDIProcessor(config_path="source/configs/data_config.yaml")
    gedi_files = list(Path(config["paths"]["gedi_dir"]).glob("*.h5"))
    if not gedi_files:
        print("⚠️ Không tìm thấy file GEDI trong thư mục raw.")
    else:
        gedi.process_gedi_files([str(f) for f in gedi_files])

    # 2. Xử lý Sentinel
    sentinel = SentinelProcessor(config_path="source/configs/data_config.yaml")
    sentinel.process_collection()
    sentinel.create_patches(image_path=config["paths"]["sentinel_dir"] + "/sentinel_image.tif")

    print("✅ Dataset prepared successfully.")

if __name__ == "__main__":
    main()
