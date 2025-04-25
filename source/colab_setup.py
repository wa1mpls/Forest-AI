import os
import ee
import subprocess
import sys

def setup_colab():
    """Setup environment for Google Colab"""
    print("🔧 Đang thiết lập môi trường Colab...")

    # Cài đặt các thư viện cần thiết nếu chưa có
    required_packages = [
        "earthengine-api",
        "geemap",
        "rasterio",
        "geopandas",
        "h5py",
        "numpy",
        "pandas",
        "tqdm",
        "pyyaml"
    ]

    for pkg in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # Tạo các thư mục cần thiết
    dirs = [
        "data/raw/sentinel",
        "data/raw/gedi",
        "data/raw/shapefiles",
        "data/processed",
        "data/outputs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Authenticate Earth Engine
    try:
        print("🔑 Đang xác thực Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize()
        print("✅ Xác thực Earth Engine thành công!")
        return True
    except Exception as e:
        print("❌ Lỗi xác thực Earth Engine:")
        print(e)
        print("👉 Hướng dẫn:")
        print("1. Truy cập https://earthengine.google.com/")
        print("2. Đăng nhập bằng tài khoản Google")
        print("3. Chạy: !earthengine authenticate")
        return False

if __name__ == "__main__":
    setup_colab()
