import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history (History): Training history object
        save_path (Path): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    for metric in history.history:
        if metric.startswith('val_') and metric != 'val_loss':
            metric_name = metric[4:]  # Remove 'val_' prefix
            plt.plot(history.history[metric], label=f'Validation {metric_name}')
            plt.plot(history.history[metric_name], label=f'Training {metric_name}')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_predictions(y_true, y_pred, save_path=None):
    """
    Plot predicted vs true values
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        save_path (Path): Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Predicted vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_names, importance_values, save_path=None):
    """
    Plot feature importance
    
    Args:
        feature_names (list): List of feature names
        importance_values (array): Importance values
        save_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Sort features by importance
    sorted_idx = np.argsort(importance_values)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.barh(pos, importance_values[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title('Feature Importance')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_attention_maps(attention_maps, save_path=None):
    """
    Plot attention maps
    
    Args:
        attention_maps (array): Attention maps
        save_path (Path): Path to save the plot
    """
    num_maps = attention_maps.shape[-1]
    fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
    
    for i in range(num_maps):
        axes[i].imshow(attention_maps[..., i], cmap='viridis')
        axes[i].set_title(f'Attention Map {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_error_distribution(y_true, y_pred, save_path=None):
    """
    Plot error distribution
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        save_path (Path): Path to save the plot
    """
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 