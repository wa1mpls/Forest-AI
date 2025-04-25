import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm

from models.vision_transformer import VisionTransformer
from utils.metrics import evaluate_model, print_metrics
from utils.visualization import plot_predictions, plot_error_distribution
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config():
    """Load configuration from YAML files"""
    config_dir = Path(__file__).resolve().parent / "configs"
    
    with open(config_dir / "data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    
    with open(config_dir / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    
    return data_config, model_config

def load_test_data(data_config):
    """Load test dataset"""
    logger.info("Loading test data...")
    
    # Load data
    data_dir = Path(data_config["paths"]["processed_data"])
    X = np.load(data_dir / "X_test.npy")
    y = np.load(data_dir / "y_test.npy")
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    return X, y

def evaluate_model_on_test(model, X_test, y_test, batch_size):
    """Evaluate model on test dataset"""
    logger.info("Evaluating model on test set...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), batch_size)):
            batch_X = X_test[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size]
            
            # Make predictions
            preds = model(batch_X)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
    
    # Calculate metrics
    metrics = evaluate_model(np.array(all_targets), np.array(all_preds))
    print_metrics(metrics)
    
    # Generate visualizations
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_predictions(np.array(all_targets), np.array(all_preds), save_path=output_dir / "predictions.png")
    plot_error_distribution(np.array(all_targets), np.array(all_preds), save_path=output_dir / "error_distribution.png")
    
    return metrics

def main():
    """Main evaluation function"""
    # Load configs
    data_config, model_config = load_config()
    
    # Load test data
    X_test, y_test = load_test_data(data_config)
    
    # Load model
    logger.info("Loading model...")
    model = VisionTransformer(model_config)
    
    # Load weights
    checkpoint_path = Path(model_config["checkpoint"]["directory"]) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        return
    
    model.load_checkpoint(checkpoint_path)
    logger.info("Model loaded successfully")
    
    # Evaluate
    metrics = evaluate_model_on_test(
        model,
        X_test,
        y_test,
        model_config["training"]["batch_size"]
    )
    
    # Save metrics
    metrics_path = Path("outputs/evaluation/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation complete. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 