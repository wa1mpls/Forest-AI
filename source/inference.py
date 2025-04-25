import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import rasterio
import json

from models.vision_transformer import VisionTransformer
from utils.logger import get_logger
from utils.visualization import plot_predictions

logger = get_logger(__name__)

def load_config():
    """Load configuration from YAML files"""
    config_dir = Path(__file__).resolve().parent / "configs"
    
    with open(config_dir / "data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    
    with open(config_dir / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    
    return data_config, model_config

def load_model(model_config, checkpoint_path):
    """Load trained model"""
    logger.info("Loading model...")
    
    model = VisionTransformer(model_config)
    
    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        return None
    
    model.load_checkpoint(checkpoint_path)
    logger.info("Model loaded successfully")
    return model

def preprocess_image(image_path, image_size):
    """Preprocess input image"""
    logger.info(f"Preprocessing image: {image_path}")
    
    # Read image
    with rasterio.open(image_path) as src:
        image = src.read()
        transform = src.transform
    
    # Normalize
    image = image / 255.0
    
    # Resize if needed
    if image.shape[1:] != image_size:
        image = torch.nn.functional.interpolate(
            torch.FloatTensor(image).unsqueeze(0),
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).numpy()
    
    return image, transform

def create_patches(image, patch_size):
    """Create patches from image"""
    patches = []
    for i in range(0, image.shape[1] - patch_size + 1, patch_size):
        for j in range(0, image.shape[2] - patch_size + 1, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def predict(model, image_path, data_config, model_config):
    """Make prediction for a single image"""
    # Preprocess image
    image_size = tuple(data_config["preprocessing"]["image_size"])
    image, transform = preprocess_image(image_path, image_size)
    
    # Create patches
    patch_size = data_config["sentinel"]["patch_size"]
    patches = create_patches(image, patch_size)
    
    # Convert to PyTorch tensors
    patches = torch.FloatTensor(np.array(patches))
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Make predictions
    logger.info("Making predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), model_config["training"]["batch_size"])):
            batch = patches[i:i+model_config["training"]["batch_size"]].to(device)
            preds = model(batch)
            predictions.extend(preds.cpu().numpy())
    
    # Create prediction map
    pred_map = np.zeros((image.shape[1], image.shape[2]))
    idx = 0
    for i in range(0, image.shape[1] - patch_size + 1, patch_size):
        for j in range(0, image.shape[2] - patch_size + 1, patch_size):
            pred_map[i:i+patch_size, j:j+patch_size] = predictions[idx]
            idx += 1
    
    return pred_map, transform

def save_prediction_map(pred_map, transform, output_path):
    """Save prediction map as GeoTIFF"""
    logger.info(f"Saving prediction map to {output_path}")
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=pred_map.shape[0],
        width=pred_map.shape[1],
        count=1,
        dtype=pred_map.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(pred_map, 1)

def main():
    """Main inference function"""
    # Load configs
    data_config, model_config = load_config()
    
    # Load model
    checkpoint_path = Path(model_config["checkpoint"]["directory"]) / "best_model.pt"
    model = load_model(model_config, checkpoint_path)
    if model is None:
        return
    
    # Get input directory
    input_dir = Path("data/inference")
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Process each image
    for image_path in input_dir.glob("*.tif"):
        try:
            # Make prediction
            pred_map, transform = predict(model, str(image_path), data_config, model_config)
            
            # Save prediction map
            output_dir = Path("outputs/inference")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_prediction.tif"
            save_prediction_map(pred_map, transform, output_path)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue
    
    logger.info("Inference complete!")

if __name__ == "__main__":
    main() 