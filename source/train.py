import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config.config import TRAIN_CONFIG, DATA_CONFIG, MODEL_DIR
from models.hybrid_forest_model import HybridForestModel
from utils.metrics import evaluate_model, print_metrics
from utils.visualization import plot_training_history, plot_predictions


def save_loss_history(history, filepath='logs/loss_history.csv'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])
        for i, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            writer.writerow([i + 1, train_loss, val_loss])


def build_callbacks():
    callbacks = []
    cb_conf = TRAIN_CONFIG['callbacks']

    if 'early_stopping' in cb_conf:
        callbacks.append(EarlyStopping(**cb_conf['early_stopping']))

    if 'reduce_lr' in cb_conf:
        callbacks.append(ReduceLROnPlateau(**cb_conf['reduce_lr']))

    if 'model_checkpoint' in cb_conf:
        callbacks.append(ModelCheckpoint(**cb_conf['model_checkpoint']))

    return callbacks


def train_model(train_dataset, val_dataset):
    # Create model
    model = HybridForestModel(
        input_shape=TRAIN_CONFIG['input_shape'],
        gedi_features=TRAIN_CONFIG['gedi_features'],
        num_classes=TRAIN_CONFIG['num_outputs']
    )
    model.build_graph(TRAIN_CONFIG['input_shape'], (TRAIN_CONFIG['gedi_features'],)).summary()

    # Compile
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=TRAIN_CONFIG['optimizer']['learning_rate'],
        beta_1=TRAIN_CONFIG['optimizer']['beta_1'],
        beta_2=TRAIN_CONFIG['optimizer']['beta_2'],
        epsilon=TRAIN_CONFIG['optimizer']['epsilon']
    )

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # Train
    history = model.fit(
        train_dataset.batch(TRAIN_CONFIG['batch_size']),
        validation_data=val_dataset.batch(TRAIN_CONFIG['batch_size']),
        epochs=TRAIN_CONFIG['num_epochs'],
        callbacks=build_callbacks()
    )

    save_loss_history(history)
    plot_training_history(history)
    return model


def load_datasets():
    import tensorflow as tf
    import pandas as pd

    def load_csv(csv_path):
        df = pd.read_csv(csv_path)
        images = []
        labels = []
        for i, row in df.iterrows():
            img_path = os.path.join(DATA_CONFIG['image_folder'], f"image_{i}.png")
            metadata_path = img_path.replace('.png', '_metadata.json')
            if os.path.exists(img_path) and os.path.exists(metadata_path):
                img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
                img = tf.image.resize(img, DATA_CONFIG['image_size']) / 255.0
                label = tf.convert_to_tensor(list(row.values), dtype=tf.float32)
                images.append(img)
                labels.append(label)
        return tf.data.Dataset.from_tensor_slices((images, labels))

    train_dataset = load_csv(DATA_CONFIG['train_csv'])
    val_dataset = load_csv(DATA_CONFIG['val_csv'])
    test_dataset = load_csv(DATA_CONFIG['test_csv'])
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Load data
    train_dataset, val_dataset, test_dataset = load_datasets()

    # Train
    model = train_model(train_dataset, val_dataset)

    # Evaluate
    print("\n🎯 Evaluating on test set...")
    y_true, y_pred = [], []
    for images, labels in test_dataset.batch(TRAIN_CONFIG['batch_size']):
        preds = model(images, training=False)
        y_true.append(labels.numpy())
        y_pred.append(preds.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    metrics = evaluate_model(y_true, y_pred)
    print_metrics(metrics)

    # Plot predictions
    plot_predictions(y_true, y_pred)
