import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from models.vision_transformer import VisionTransformer
from utils.metrics import evaluate_model, print_metrics
from utils.visualization import plot_training_history, plot_predictions
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

def load_data(data_config, model_config):
    """Load training data"""
    logger.info("Loading training data...")
    
    # Load data
    data_dir = Path(data_config["paths"]["processed_data"])
    X = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=data_config["split"]["random_seed"]
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config["training"]["batch_size"],
        shuffle=False
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, model_config):
    """Train the model"""
    logger.info("Training model...")
    
    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_config["training"]["optimizer"]["learning_rate"],
        weight_decay=model_config["training"]["optimizer"]["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=model_config["training"]["num_epochs"]
    )
    
    # Training loop
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    for epoch in range(model_config["training"]["num_epochs"]):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config['training']['num_epochs']}"):
            batch = [b.to(device) for b in batch]
            loss = model.train_step(batch, optimizer, criterion)
            train_losses.append(loss)
        
        train_loss = np.mean(train_losses)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            batch = [b.to(device) for b in batch]
            loss, preds, targets = model.validate_step(batch, criterion)
            val_losses.append(loss)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        val_loss = np.mean(val_losses)
        history["val_loss"].append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                Path(model_config["checkpoint"]["directory"]) / "best_model.pt"
            )
        
        # Log progress
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        # Calculate metrics
        metrics = evaluate_model(np.array(all_targets), np.array(all_preds))
        print_metrics(metrics)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def main():
    """Main training function"""
    # Load configs
    data_config, model_config = load_config()
    
    # Load data
    train_loader, val_loader = load_data(data_config, model_config)
    
    # Create model
    model = VisionTransformer(model_config)
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, model_config)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()