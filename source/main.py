import os
import argparse
import yaml
from pathlib import Path
import sys

import ee
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
        logger.error(f"❌ Lỗi khi đọc file config {config_path}: {str(e)}")
        sys.exit(1)

def prepare_data(config):
    """Prepare data for training"""
    logger.info("🔧 Bắt đầu xử lý dữ liệu...")

    # Xử lý Sentinel-2
    sentinel = SentinelProcessor(config)
    if not sentinel.process_collection():
        logger.error("❌ Lỗi xử lý dữ liệu Sentinel-2")
        return False

    # Xử lý GEDI
    gedi = GEDIProcessor(config)
    gedi_files = list(Path(config["paths"]["gedi_dir"]).glob("*.h5"))
    if not gedi.process_gedi_files(gedi_files):
        logger.error("❌ Lỗi xử lý dữ liệu GEDI")
        return False

    # Tạo patch dataset
    if not gedi.create_training_dataset(config["paths"]["patches_dir"]):
        logger.error("❌ Lỗi tạo tập dữ liệu huấn luyện")
        return False

    logger.info("✅ Hoàn tất chuẩn bị dữ liệu!")
    return True

def initialize_earth_engine():
    """Khởi tạo Earth Engine nếu đã xác thực từ trước"""
    try:
        ee.Initialize()
        logger.info("🌍 Google Earth Engine đã sẵn sàng")
    except Exception as e:
        logger.warning("⚠️ Không thể khởi tạo Earth Engine.")
        logger.warning("👉 Bạn cần chạy lệnh sau trong Colab để xác thực:")
        logger.warning("!earthengine authenticate")
        logger.warning(e)

def run_pipeline(mode, config_path):
    """Chạy toàn bộ pipeline theo mode"""
    logger.info(f"🚀 Khởi động pipeline với chế độ: {mode}")

    config = load_config(config_path)

    if mode == "all" or mode == "download":
        logger.info("📥 Đang tải dữ liệu...")
        download_dataset()

    if mode == "all" or mode == "prepare":
        logger.info("🛠️ Đang xử lý dữ liệu...")
        if not prepare_data(config):
            logger.error("❌ Dừng pipeline do lỗi xử lý dữ liệu.")
            return

    if mode == "all" or mode == "train":
        logger.info("🎯 Đang huấn luyện mô hình...")
        train_model(config)

    if mode == "all" or mode == "evaluate":
        logger.info("📊 Đang đánh giá mô hình...")
        evaluate_model(config)

    if mode == "all" or mode == "inference":
        logger.info("🔎 Đang suy luận mô hình...")
        run_inference(config)

def main():
    args = parse_args()

    # Kiểm tra nếu đang chạy trong Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        initialize_earth_engine()

    run_pipeline(args.mode, args.config)

if __name__ == "__main__":
    main()
