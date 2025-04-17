import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, title="Ground Truth vs Predictions"):
    """Plot predictions against ground truth"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()

def plot_loss_history(train_losses, val_losses):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plot feature importance from model"""
    # Get feature importance from model
    importance = model.regressor[0].weight.detach().cpu().numpy()
    importance = np.abs(importance).mean(axis=0)
    
    # Sort features by importance
    idx = np.argsort(importance)
    importance = importance[idx]
    feature_names = [feature_names[i] for i in idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance)
    plt.yticks(range(len(importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from Model')
    plt.show()

def plot_attention_maps(model, image):
    """Plot attention maps from model"""
    # Get attention maps
    attention = model.spectral_attention.attention(image)
    
    # Plot
    plt.figure(figsize=(15, 5))
    for i in range(attention.shape[1]):
        plt.subplot(1, attention.shape[1], i+1)
        plt.imshow(attention[0,i].detach().cpu().numpy())
        plt.title(f'Attention Map {i+1}')
        plt.axis('off')
    plt.show() 