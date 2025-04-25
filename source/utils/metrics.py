import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def r2_score_tf(y_true, y_pred):
    """
    R-squared (coefficient of determination)
    """
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

def bias(y_true, y_pred):
    """
    Mean Bias
    """
    return tf.reduce_mean(y_pred - y_true)

def evaluate_model(y_true, y_pred):
    """
    Evaluate using NumPy-based metrics (for final reporting)
    
    Args:
        y_true: array-like (numpy array or tensor)
        y_pred: array-like

    Returns:
        dict of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true)
    }

def print_metrics(metrics_dict):
    """
    Print dictionary of metric values in readable format
    """
    print("\n📊 Model Evaluation Metrics:")
    print("-" * 30)
    for k, v in metrics_dict.items():
        print(f"{k:>6}: {v:.4f}")
    print("-" * 30)
