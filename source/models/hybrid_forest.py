import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------- SpectralAttention ----------------------
class SpectralAttention(Model):
    def __init__(self, num_bands, reduction_ratio=8):
        super(SpectralAttention, self).__init__()
        self.num_bands = num_bands
        self.reduction_ratio = reduction_ratio

        self.channel_attention = self._build_channel_attention()
        self.spatial_attention = self._build_spatial_attention()

    def _build_channel_attention(self):
        inputs = layers.Input(shape=(self.num_bands,))
        x = layers.Dense(self.num_bands // self.reduction_ratio, activation='relu')(inputs)
        x = layers.Dense(self.num_bands, activation='sigmoid')(x)
        return Model(inputs, x)

    def _build_spatial_attention(self):
        inputs = layers.Input(shape=(None, None, 1))
        x = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(inputs)
        return Model(inputs, x)

    def call(self, inputs):
        channel_avg = tf.reduce_mean(inputs, axis=[1, 2])
        channel_weights = self.channel_attention(channel_avg)
        channel_weights = tf.expand_dims(tf.expand_dims(channel_weights, 1), 1)
        channel_out = inputs * channel_weights

        spatial_avg = tf.reduce_mean(channel_out, axis=-1, keepdims=True)
        spatial_weights = self.spatial_attention(spatial_avg)
        return channel_out * spatial_weights

# ---------------------- EnhancedFeatures ----------------------
class EnhancedFeatures(Model):
    def __init__(self, num_bands):
        super(EnhancedFeatures, self).__init__()
        self.num_bands = num_bands

        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')

        self.attention = SpectralAttention(num_bands=256)
        self.fusion = layers.Conv2D(128, 1, padding='same', activation='relu')

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = self.attention(x3)

        x = tf.concat([x1, x2, x3], axis=-1)
        return self.fusion(x)

# ---------------------- HybridForestModel ----------------------
class HybridForestModel(Model):
    def __init__(self, input_shape, gedi_features, num_classes):
        super(HybridForestModel, self).__init__()
        self.image_branch = self._build_image_branch(input_shape)
        self.gedi_branch = self._build_gedi_branch(gedi_features)
        self.combined_dense = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_classes)

    def _build_image_branch(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = EnhancedFeatures(num_bands=input_shape[-1])(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        return Model(inputs, x)

    def _build_gedi_branch(self, gedi_features):
        inputs = layers.Input(shape=(gedi_features,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        return Model(inputs, x)

    def call(self, inputs):
        images, gedi_features = inputs
        image_features = self.image_branch(images)
        gedi_features = self.gedi_branch(gedi_features)

        combined = tf.concat([image_features, gedi_features], axis=1)
        x = self.combined_dense(combined)
        return self.output_layer(x)

    def build_graph(self, image_shape, gedi_shape):
        image_input = layers.Input(shape=image_shape)
        gedi_input = layers.Input(shape=gedi_shape)
        return Model(inputs=[image_input, gedi_input], outputs=self.call([image_input, gedi_input]))
