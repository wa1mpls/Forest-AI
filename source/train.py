import os
import csv
import numpy as np
import tensorflow as tf
import yaml
import pandas as pd
from pathlib import Path

from models.hybrid_forest_model import HybridForestModel
from utils.metrics import evaluate_model, print_metrics
from utils.visualization import plot_training_history, plot_predictions

# --- Load config from YAML ---
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

with open(CONFIG_DIR / "data_config.yaml") as f:
    DATA_CONFIG = yaml.safe_load(f)

with open(CONFIG_DIR / "model_config.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

TRAIN_CONFIG = MODEL_CONFIG["training"]
IMAGE_SIZE = tuple(DATA_CONFIG["preprocessing"]["image_size"])
BATCH_SIZE = TRAIN_CONFIG["batch_size"]


# Save training loss history
def save_loss_history(history, filepath='logs/loss_history.csv'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
        for i, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            writer.writerow([i + 1, train_loss, val_loss])


# Create training callbacks
def build_callbacks():
    cb_conf = TRAIN_CONFIG.get("early_stopping", {})
    model_ckpt_dir = TRAIN_CONFIG.get("save_dir", "checkpoints/")
    os.makedirs(model_ckpt_dir, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=cb_conf.get("monitor", "val_loss"),
            patience=cb_conf.get("patience", 5),
            restore_best_weights=cb_conf.get("restore_best_weights", True),
            min_delta=cb_conf.get("min_delta", 0.001),
            verbose=cb_conf.get("verbose", True)
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_ckpt_dir, "best_model.keras"),
            save_best_only=True,
            save_weights_only=cb_conf.get("save_weights_only", True),
            monitor=cb_conf.get("monitor", "val_loss"),
            mode=cb_conf.get("mode", "min")
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]


# Build and train model
def train_model(train_dataset, val_dataset):
    model = HybridForestModel(
        input_shape=TRAIN_CONFIG["input_shape"],
        gedi_features=TRAIN_CONFIG["gedi_features"],
        num_classes=TRAIN_CONFIG["num_outputs"]
    )
    model.build_graph(TRAIN_CONFIG["input_shape"], (TRAIN_CONFIG["gedi_features"],)).summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=TRAIN_CONFIG["optimizer"]["learning_rate"],
        beta_1=TRAIN_CONFIG["optimizer"]["beta1"],
        beta_2=TRAIN_CONFIG["optimizer"]["beta2"],
        epsilon=TRAIN_CONFIG["optimizer"]["epsilon"]
    )

    model.compile(optimizer=optimizer, loss='mse', metrics=["mae"])

    history = model.fit(
        train_dataset.batch(BATCH_SIZE),
        validation_data=val_dataset.batch(BATCH_SIZE),
        epochs=TRAIN_CONFIG["num_epochs"],
        callbacks=build_callbacks()
    )

    save_loss_history(history)
    plot_training_history(history)
    return model


# Load datasets from CSV + images
def load_datasets():
    def load_csv(csv_path):
        df = pd.read_csv(DATA_CONFIG["train_csv"])
        images, labels = [], []
        for i, row in df.iterrows():
            img_path = os.path.join(DATA_CONFIG['paths']['image_folder'], f"image_{i}.png")
            metadata_path = img_path.replace(".png", "_metadata.json")
            if os.path.exists(img_path) and os.path.exists(metadata_path):
                img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
                img = tf.image.resize(img, IMAGE_SIZE) / 255.0
                label = tf.convert_to_tensor(list(row.values), dtype=tf.float32)
                images.append(img)
                labels.append(label)
        return tf.data.Dataset.from_tensor_slices((images, labels))

    return (
        train_dataset = load_csv(DATA_CONFIG['paths']['train_csv'])
        val_dataset = load_csv(DATA_CONFIG['paths']['val_csv'])
        test_dataset = load_csv(DATA_CONFIG['paths']['test_csv'])

    )


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_datasets()
    model = train_model(train_dataset, val_dataset)

    print("\n🎯 Evaluating on test set...")
    y_true, y_pred = [], []
    for images, labels in test_dataset.batch(BATCH_SIZE):
        preds = model(images, training=False)
        y_true.append(labels.numpy())
        y_pred.append(preds.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    metrics = evaluate_model(y_true, y_pred)
    print_metrics(metrics)
    plot_predictions(y_true, y_pred)
