# Forest AI - Biomass Estimation using GEDI and Sentinel-2

This project combines GEDI L4A, L2A, L2B datasets with Sentinel-2 imagery to estimate forest aboveground biomass density (AGBD).

## Features

- Combines multiple GEDI datasets (L4A, L2A, L2B) for comprehensive forest analysis
- Uses Sentinel-2 imagery for spectral information
- Implements hybrid CNN-ViT architecture for biomass estimation
- Includes spectral attention and enhanced feature extraction
- Provides comprehensive evaluation metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wa1mpls/Forest-AI
cd forest-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Earth Engine:
```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

## Data Structure

- `data/`: Contains data processing and dataset classes
  - `download.py`: Functions to download GEDI and Sentinel-2 data
  - `preprocessing.py`: Data preprocessing utilities
  - `dataset.py`: PyTorch dataset implementation

- `models/`: Contains model architectures
  - `spectral_attention.py`: Spectral attention module
  - `enhanced_features.py`: Enhanced feature extraction
  - `hybrid_forest.py`: Main hybrid model

- `utils/`: Utility functions
  - `visualization.py`: Plotting and visualization
  - `metrics.py`: Evaluation metrics

## Usage

1. Download data:
```python
from data.download import download_gedi_data
download_gedi_data(region, date_range)
```

2. Train model:
```python
from train import train_model
model = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50
)
```

3. Evaluate:
```python
from utils.metrics import evaluate_model
metrics = evaluate_model(model, test_loader)
```

## Data Sources

- GEDI L4A: Aboveground Biomass Density (AGBD)
- GEDI L2A: Canopy Height Metrics
- GEDI L2B: Canopy Cover and PAI
- Sentinel-2: Spectral Imagery

## License

MIT License

## Contact

Your Name - ngonguyenthanhthanh00@gmail.com
Project Link: https://github.com/wa1mpls/Forest-AI