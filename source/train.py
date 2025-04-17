import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.dataset import get_dataloaders
from models.hybrid_forest import HybridForestModel
from utils.metrics import TotalLoss, evaluate_model
from utils.visualization import plot_loss_history, plot_predictions
from config import MODEL_CONFIG, DATA_CONFIG
import numpy as np

def train_model(train_loader, val_loader, num_epochs=None, device=None):
    if num_epochs is None:
        num_epochs = MODEL_CONFIG['num_epochs']
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = HybridForestModel().to(device)
    
    # Initialize loss function and optimizer
    criterion = TotalLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = MODEL_CONFIG['patience']
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Plot training history
    plot_loss_history(train_losses, val_losses)
    
    return model

if __name__ == "__main__":
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df="data/train.csv",
        val_df="data/val.csv",
        test_df="data/test.csv",
        image_folder="data/images"
    )
    
    # Train model
    model = train_model(train_loader, val_loader)
    
    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, test_loader, device)
    print("Test Metrics:", metrics)
    
    # Plot predictions
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    plot_predictions(all_labels, all_preds) 