import tensorflow as tf
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    
    Args:
        y_true (tensor): True values
        y_pred (tensor): Predicted values
        
    Returns:
        tensor: RMSE value
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    
    Args:
        y_true (tensor): True values
        y_pred (tensor): Predicted values
        
    Returns:
        tensor: MAE value
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def r2_score_tf(y_true, y_pred):
    """
    R-squared score
    
    Args:
        y_true (tensor): True values
        y_pred (tensor): Predicted values
        
    Returns:
        tensor: R2 score
    """
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

def bias(y_true, y_pred):
    """
    Bias
    
    Args:
        y_true (tensor): True values
        y_pred (tensor): Predicted values
        
    Returns:
        tensor: Bias value
    """
    return tf.reduce_mean(y_pred - y_true)

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance with multiple metrics
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        dict: Dictionary of metric values
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true)
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print evaluation metrics
    
    Args:
        metrics (dict): Dictionary of metric values
    """
    print("\nModel Evaluation Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 30)

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