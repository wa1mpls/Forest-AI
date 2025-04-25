import unittest
import tensorflow as tf
import numpy as np
from source.models.hybrid_forest_model import HybridForestModel

class TestHybridForestModelTF(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.height = 224
        self.width = 224
        self.channels = 10  # ví dụ Sentinel-2: 6 bands + 3 chỉ số + 1 mask
        self.gedi_features = 6
        self.num_classes = 1

        self.model = HybridForestModel(
            input_shape=(self.height, self.width, self.channels),
            gedi_features=self.gedi_features,
            num_classes=self.num_classes
        )

        self.images = tf.random.normal((self.batch_size, self.height, self.width, self.channels))
        self.gedi = tf.random.normal((self.batch_size, self.gedi_features))

    def test_forward_pass(self):
        output = self.model((self.images, self.gedi))
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_output_dtype(self):
        output = self.model((self.images, self.gedi))
        self.assertEqual(output.dtype, tf.float32)

    def test_gradient_flow(self):
        with tf.GradientTape() as tape:
            output = self.model((self.images, self.gedi), training=True)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        for grad in gradients:
            self.assertIsNotNone(grad)

    def test_train_step(self):
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.MeanSquaredError()

        y_true = tf.random.normal((self.batch_size, self.num_classes))

        with tf.GradientTape() as tape:
            y_pred = self.model((self.images, self.gedi), training=True)
            loss = loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.assertGreaterEqual(loss.numpy(), 0.0)

    def test_save_and_load(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            self.model.save(f"{tmpdir}/model")

            # Load model
            loaded_model = tf.keras.models.load_model(f"{tmpdir}/model", custom_objects={
                "HybridForestModel": HybridForestModel
            })

            # Compare output
            output1 = self.model((self.images, self.gedi))
            output2 = loaded_model((self.images, self.gedi))

            np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-5)

    def test_device(self):
        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(device):
            model = HybridForestModel(
                input_shape=(self.height, self.width, self.channels),
                gedi_features=self.gedi_features,
                num_classes=self.num_classes
            )
            output = model((self.images, self.gedi))
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

if __name__ == '__main__':
    unittest.main()
