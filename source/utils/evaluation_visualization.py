import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Tạo thư mục lưu trữ kết quả nếu chưa tồn tại
os.makedirs('evaluation_results', exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """
    Vẽ confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix.png')
    plt.close()

def plot_metrics(metrics_dict, title='Model Evaluation Metrics'):
    """
    Vẽ biểu đồ các metrics
    """
    plt.figure(figsize=(12, 6))
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.plot(kind='bar')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evaluation_results/metrics.png')
    plt.close()

def plot_roc_curve(fpr, tpr, auc, title='ROC Curve'):
    """
    Vẽ đường ROC
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('evaluation_results/roc_curve.png')
    plt.close()

def save_classification_report(y_true, y_pred, classes):
    """
    Lưu classification report
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('evaluation_results/classification_report.csv')
    
    # Vẽ biểu đồ precision, recall, f1-score
    metrics = ['precision', 'recall', 'f1-score']
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(classes, [report[cls][metric] for cls in classes], 
                marker='o', label=metric)
    plt.title('Classification Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evaluation_results/classification_metrics.png')
    plt.close()

def plot_learning_curve(train_sizes, train_scores, val_scores, title='Learning Curve'):
    """
    Vẽ learning curve
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation_results/learning_curve.png')
    plt.close()

# Ví dụ sử dụng các hàm trên
if __name__ == "__main__":
    # Dữ liệu mẫu
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    classes = ['Class 0', 'Class 1']
    
    # Vẽ confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes)
    
    # Vẽ metrics
    metrics = {
        'Accuracy': [0.8],
        'Precision': [0.75],
        'Recall': [0.8],
        'F1-Score': [0.77]
    }
    plot_metrics(metrics)
    
    # Vẽ ROC curve
    fpr = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 0.8, 0.9, 1.0])
    auc = 0.85
    plot_roc_curve(fpr, tpr, auc)
    
    # Lưu classification report
    save_classification_report(y_true, y_pred, classes)
    
    # Vẽ learning curve
    train_sizes = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    train_scores = np.array([[0.6, 0.65, 0.7], [0.7, 0.72, 0.75], 
                            [0.75, 0.78, 0.8], [0.8, 0.82, 0.85],
                            [0.85, 0.87, 0.9], [0.9, 0.92, 0.95]])
    val_scores = np.array([[0.5, 0.55, 0.6], [0.6, 0.62, 0.65],
                          [0.65, 0.68, 0.7], [0.7, 0.72, 0.75],
                          [0.75, 0.77, 0.8], [0.8, 0.82, 0.85]])
    plot_learning_curve(train_sizes, train_scores, val_scores) 