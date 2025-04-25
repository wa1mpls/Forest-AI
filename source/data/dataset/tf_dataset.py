import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import rasterio
from sklearn.model_selection import train_test_split

class ForestTFDataset:
    def __init__(self, sentinel_dir, gedi_dir, config):
        self.sentinel_dir = Path(sentinel_dir)
        self.gedi_dir = Path(gedi_dir)
        self.config = config
        self.sentinel_data = self._load_sentinel_data()
        self.gedi_data = self._load_gedi_data()

    def _load_sentinel_data(self):
        sentinel_data = {}
        for band in self.config["sentinel_bands"]:
            band_path = self.sentinel_dir / f"{band}.tif"
            if band_path.exists():
                with rasterio.open(band_path) as src:
                    data = src.read(1)
                    if self.config["normalize"]:
                        data = (data - np.min(data)) / (np.max(data) - np.min(data))
                    sentinel_data[band] = data

        if self.config["compute_ndvi"] and "B4" in sentinel_data and "B8" in sentinel_data:
            ndvi = (sentinel_data["B8"] - sentinel_data["B4"]) / (sentinel_data["B8"] + sentinel_data["B4"])
            sentinel_data["NDVI"] = ndvi

        return sentinel_data

    def _load_gedi_data(self):
        gedi_files = list(self.gedi_dir.glob("*.csv"))
        if not gedi_files:
            raise FileNotFoundError(f"No GEDI data found in {self.gedi_dir}")
        gedi_data = pd.concat([pd.read_csv(f) for f in gedi_files])
        return gedi_data

    def split_data(self, train_ratio, val_ratio, test_ratio, random_seed=42):
        X = np.stack([self.sentinel_data[band] for band in self.config["sentinel_bands"]], axis=-1)
        y = self.gedi_data["agbd"].values

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1-train_ratio, random_state=random_seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_size, random_state=random_seed
        )

        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        return train_data, val_data, test_data
