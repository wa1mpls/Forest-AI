# Forest-AI: Hệ thống ước lượng mật độ carbon tại rừng Amazon

Hệ thống sử dụng xử lý ảnh vệ tinh và học sâu để ước lượng mật độ carbon (AGB - Above Ground Biomass) tại rừng Amazon.

## Mục đích

- Tự động hóa việc kiểm kê sinh khối và trữ lượng carbon tại các vùng rừng sâu
- Hỗ trợ đánh giá chính sách REDD+, biến đổi khí hậu, quản lý rừng
- Thay thế phần nào khảo sát thực địa thủ công

## Cấu trúc dự án

```
forest-ai/
├── data/                  # Dữ liệu
├── src/                   # Source code
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── configs/              # Cấu hình
├── utils/                # Tiện ích
└── scripts/              # Scripts
```

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/yourusername/forest-ai.git
cd forest-ai
```

2. Tạo môi trường ảo và cài đặt dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. Cài đặt Google Earth Engine:
```bash
earthengine authenticate
```

## Sử dụng

1. Tải dữ liệu:
```bash
python scripts/download_data.py
```

2. Tiền xử lý dữ liệu:
```bash
python scripts/preprocess.py
```

3. Huấn luyện mô hình:
```bash
python src/training/train.py
```

4. Dự đoán:
```bash
python scripts/predict.py
```

## Dữ liệu

- Sentinel-2: Ảnh vệ tinh đa phổ
- GEDI L4A: Dữ liệu đo sinh khối từ laser vệ tinh

## Mô hình

- Vision Transformer (ViT) với Spectral Attention
- Đầu vào: Patch ảnh 16x16 pixel
- Đầu ra: Giá trị AGB (Mg/ha)

## Đánh giá

- R²: 0.72
- MAE: 11.3 Mg/ha
- RMSE: 15.7 Mg/ha

## Hạn chế

- Phụ thuộc vào chất lượng ảnh vệ tinh
- Cần dữ liệu GEDI chất lượng cao
- Yêu cầu GPU để huấn luyện

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## Giấy phép

MIT License 