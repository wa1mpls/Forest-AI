# source/prepare_dataset.py

import os
import yaml
import pandas as pd
from pathlib import Path
from data.preprocessing.gedi_preprocessing import GEDIProcessor
from data.preprocessing.sentinel_preprocessing import SentinelProcessor

def load_config():
    config_path = Path(__file__).resolve().parent / "configs" / "data_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config, str(config_path)

def main():
    # 🧩 Load config và xác định đúng đường dẫn
    config, config_path_str = load_config()

    # 🧩 1. Xử lý GEDI
    print("🔍 Đang xử lý dữ liệu GEDI...")
    gedi = GEDIProcessor(config_path=config_path_str)
    gedi_files = list(Path(config["paths"]["gedi_dir"]).glob("*.h5"))
    if not gedi_files:
        print("⚠️ Không tìm thấy file GEDI (.h5) trong thư mục:", config["paths"]["gedi_dir"])
    else:
        gedi.process_gedi_files([str(f) for f in gedi_files])

    # 🧩 2. Xử lý Sentinel-2
    print("🛰️ Đang xử lý ảnh Sentinel-2...")
    sentinel = SentinelProcessor(config_path=config_path_str)
    sentinel.process_collection()
    image_path = Path(config["paths"]["sentinel_dir"]) / "sentinel_image.tif"
    sentinel.create_patches(image_path=str(image_path))

    print("✅ Dataset prepared successfully!")

if __name__ == "__main__":
    main()
