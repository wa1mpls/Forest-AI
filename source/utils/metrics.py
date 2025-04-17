import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate various regression metrics"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataloader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return calculate_metrics(all_labels, all_preds)

class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        
        # Add losses for different features
        spectral_loss = self.l1_loss(pred[:,:3], target[:,:3])
        structural_loss = self.l1_loss(pred[:,3:], target[:,3:])
        
        return mse_loss + 0.5*l1_loss + 0.3*spectral_loss + 0.2*structural_loss 