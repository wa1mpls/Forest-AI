import os
import ee
import subprocess
import sys

def setup_colab():
    """Setup environment for Google Colab sau khi đã xác thực GEE"""
    print("🔧 Đang thiết lập môi trường Colab...")

    # Cài đặt các thư viện cần thiết nếu chưa có
    ee.Initialize(project='ee-ngonguyenthanhthanh00')  # ✅


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

    # Initialize Earth Engine
    try:
        print("🚀 Đang khởi tạo Google Earth Engine...")
        ee.Initialize()
        print("✅ Earth Engine đã sẵn sàng!")
        return True
    except Exception as e:
        print("❌ Không thể khởi tạo Earth Engine:")
        print(e)
        print("👉 Có thể bạn chưa chạy xác thực. Hãy chạy lệnh này trước trong Colab:")
        print("!earthengine authenticate")
        return False

if __name__ == "__main__":
    setup_colab()
