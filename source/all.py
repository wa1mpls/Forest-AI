import os
import ee
import geemap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Thiết lập môi trường"""
    logger.info("🔧 Đang thiết lập môi trường...")
    
    # Cài đặt các thư viện cần thiết
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
        os.system(f"pip install {pkg}")
    
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
    
    # Xác thực Earth Engine
    try:
        logger.info("🔑 Đang xác thực Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize()
        logger.info("✅ Xác thực Earth Engine thành công!")
        return True
    except Exception as e:
        logger.error("❌ Lỗi xác thực Earth Engine:")
        logger.error(e)
        logger.error("👉 Hướng dẫn:")
        logger.error("1. Truy cập https://earthengine.google.com/")
        logger.error("2. Đăng nhập bằng tài khoản Google")
        logger.error("3. Chạy: !earthengine authenticate")
        return False

def load_config():
    """Đọc cấu hình từ file YAML"""
    config_path = Path(__file__).resolve().parent / "configs" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_data(config):
    """Lấy dữ liệu từ Google Earth Engine"""
    logger.info("📥 Đang lấy dữ liệu từ Earth Engine...")
    
    # Định nghĩa vùng quan tâm
    region = ee.Geometry.Rectangle([
        config["region"]["bounds"]["min_lon"],
        config["region"]["bounds"]["min_lat"],
        config["region"]["bounds"]["max_lon"],
        config["region"]["bounds"]["max_lat"]
    ])
    
    # Định nghĩa khoảng thời gian
    start_date = ee.Date(config["date_range"]["start"])
    end_date = ee.Date(config["date_range"]["end"])
    
    # Lấy dữ liệu Sentinel-2
    sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .sort('CLOUD_COVERAGE_ASSESSMENT') \
        .first()
    
    # Tính toán các chỉ số phổ
    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
    gndvi = sentinel.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    nbr = sentinel.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # Kết hợp các đặc trưng
    features = sentinel.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
        .addBands([ndvi, gndvi, nbr])
    
    # Lấy dữ liệu GEDI
    gedi = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .select(['agbd', 'agbd_se', 'l2_quality_flag', 'sensitivity', 'degrade_flag'])
    
    # Lọc dữ liệu GEDI
    gedi = gedi.filter(ee.Filter.gt('l2_quality_flag', 0)) \
        .filter(ee.Filter.gt('sensitivity', 0.95)) \
        .filter(ee.Filter.eq('degrade_flag', 0))
    
    return features, gedi, region

def create_patches(features, gedi, config):
    """Tạo các patch và gán nhãn GEDI"""
    logger.info("🔄 Đang tạo patches...")
    
    patch_size = config["preprocessing"]["image_size"][0]
    num_features = config["preprocessing"]["num_features"]
    
    # Lấy kích thước ảnh
    image_info = features.getInfo()
    height = image_info['bands'][0]['dimensions'][0]
    width = image_info['bands'][0]['dimensions'][1]
    
    # Tính số lượng patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    patches = []
    labels = []
    patch_info = []
    
    for i in tqdm(range(num_patches_h)):
        for j in range(num_patches_w):
            # Trích xuất patch
            patch = features.sample(
                region=ee.Geometry.Rectangle([
                    j*patch_size, i*patch_size,
                    (j+1)*patch_size, (i+1)*patch_size
                ]),
                scale=10
            )
            
            # Lấy điểm GEDI trong patch
            gedi_points = gedi.sample(
                region=ee.Geometry.Rectangle([
                    j*patch_size, i*patch_size,
                    (j+1)*patch_size, (i+1)*patch_size
                ]),
                scale=10
            )
            
            # Lấy dữ liệu patch
            patch_data = patch.getInfo()
            gedi_data = gedi_points.getInfo()
            
            if len(gedi_data['features']) >= config["preprocessing"]["min_gedi_points"]:
                # Trích xuất đặc trưng
                patch_features = np.array([f['properties'] for f in patch_data['features']])
                patch_features = patch_features.reshape(patch_size, patch_size, -1)
                
                # Trích xuất nhãn GEDI
                gedi_labels = np.array([f['properties']['agbd'] for f in gedi_data['features']])
                
                patches.append(patch_features)
                labels.append(np.mean(gedi_labels))
                
                # Lưu thông tin patch
                patch_info.append({
                    'patch_id': len(patches) - 1,
                    'x': j*patch_size,
                    'y': i*patch_size,
                    'num_gedi_points': len(gedi_data['features']),
                    'mean_agbd': np.mean(gedi_labels),
                    'std_agbd': np.std(gedi_labels)
                })
    
    return np.array(patches), np.array(labels), pd.DataFrame(patch_info)

def save_data(patches, labels, patch_info, config):
    """Lưu dữ liệu đã xử lý"""
    logger.info("💾 Đang lưu dữ liệu...")
    
    output_dir = Path(config["paths"]["processed_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "patches.npy", patches)
    np.save(output_dir / "labels.npy", labels)
    patch_info.to_csv(output_dir / "patch_info.csv", index=False)
    
    logger.info(f"✅ Đã tạo {len(patches)} patches với shape {patches.shape}")
    logger.info(f"✅ Labels shape: {labels.shape}")
    logger.info(f"✅ Average AGB per patch: {np.mean(labels):.2f} Mg/ha")
    logger.info(f"✅ Standard deviation of AGB: {np.std(labels):.2f} Mg/ha")

class ForestModel(nn.Module):
    """Mô hình dự đoán sinh khối rừng"""
    def __init__(self, input_channels=10):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 2 * 2)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def prepare_data_for_training(config):
    """Chuẩn bị dữ liệu cho huấn luyện"""
    logger.info("📊 Đang chuẩn bị dữ liệu cho huấn luyện...")
    
    # Đọc dữ liệu
    data_dir = Path(config["paths"]["processed_data"])
    patches = np.load(data_dir / "patches.npy")
    labels = np.load(data_dir / "labels.npy")
    
    # Chia dữ liệu
    X_train, X_val, y_train, y_val = train_test_split(
        patches, labels,
        test_size=0.2,
        random_state=config["split"]["random_seed"]
    )
    
    # Chuyển đổi sang PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Tạo datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["preprocessing"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["preprocessing"]["batch_size"],
        shuffle=False
    )
    
    return train_loader, val_loader

def train_model(train_loader, val_loader, config):
    """Huấn luyện mô hình"""
    logger.info("🎯 Đang huấn luyện mô hình...")
    
    # Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForestModel().to(device)
    
    # Loss function và optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(config["training"]["num_epochs"]):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch[0])
            loss = criterion(outputs, batch[1])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                outputs = model(batch[0])
                loss = criterion(outputs, batch[1])
                
                val_losses.append(loss.item())
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch[1].cpu().numpy())
        
        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(config["paths"]["outputs"]) / "best_model.pt")
        
        # Log progress
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
    
    return model, history

def evaluate_model(model, val_loader, config):
    """Đánh giá mô hình"""
    logger.info("📈 Đang đánh giá mô hình...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = [b.to(device) for b in batch]
            outputs = model(batch[0])
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch[1].cpu().numpy())
    
    # Tính toán metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  RMSE: {rmse:.4f} Mg/ha")
    logger.info(f"  R²: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'r2': r2
    }

def main():
    """Hàm chính"""
    # Thiết lập môi trường
    if not setup_environment():
        return
    
    # Đọc cấu hình
    config = load_config()
    
    # Lấy dữ liệu
    features, gedi, region = get_data(config)
    
    # Tạo patches
    patches, labels, patch_info = create_patches(features, gedi, config)
    
    # Lưu dữ liệu
    save_data(patches, labels, patch_info, config)
    
    # Chuẩn bị dữ liệu cho huấn luyện
    train_loader, val_loader = prepare_data_for_training(config)
    
    # Huấn luyện mô hình
    model, history = train_model(train_loader, val_loader, config)
    
    # Đánh giá mô hình
    metrics = evaluate_model(model, val_loader, config)
    
    logger.info("🎉 Hoàn thành pipeline!")

if __name__ == "__main__":
    main() 