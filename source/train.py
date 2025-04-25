import os
import csv
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from data.dataset import get_dataloaders
from models.hybrid_forest import HybridForestModel
from utils.metrics import TotalLoss, evaluate_model
from utils.visualization import plot_loss_history, plot_predictions
from config import MODEL_CONFIG, DATA_CONFIG

def save_loss_history(train_losses, val_losses, filepath='logs/loss_history.csv'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
        for i, (t, v) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i + 1, t, v])

def train_model(train_loader, val_loader, num_epochs=None, device=None):
    num_epochs = num_epochs or MODEL_CONFIG['num_epochs']
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridForestModel().to(device)
    criterion = TotalLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=MODEL_CONFIG['learning_rate'],
                            weight_decay=MODEL_CONFIG['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = MODEL_CONFIG['patience']
    counter = 0

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ----- Early Stopping -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model and save loss history
    model.load_state_dict(torch.load("best_model.pth"))
    save_loss_history(train_losses, val_losses)
    plot_loss_history(train_losses, val_losses)
    return model

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df=DATA_CONFIG['train_csv'],
        val_df=DATA_CONFIG['val_csv'],
        test_df=DATA_CONFIG['test_csv'],
        image_folder=DATA_CONFIG['image_folder']
    )

    model = train_model(train_loader, val_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, test_loader, device)
    print("Test Metrics:", metrics)

    # Plot prediction on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    plot_predictions(all_labels, all_preds)
