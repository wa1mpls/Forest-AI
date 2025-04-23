import tensorflow as tf
from tensorflow.keras import layers, Model

class SpectralAttention(Model):
    def __init__(self, num_bands, reduction_ratio=8):
        """
        Initialize spectral attention module
        
        Args:
            num_bands (int): Number of spectral bands
            reduction_ratio (int): Reduction ratio for channel attention
        """
        super(SpectralAttention, self).__init__()
        
        self.num_bands = num_bands
        self.reduction_ratio = reduction_ratio
        
        # Channel attention
        self.channel_attention = self._build_channel_attention()
        
        # Spatial attention
        self.spatial_attention = self._build_spatial_attention()
        
    def _build_channel_attention(self):
        """
        Build channel attention branch
        
        Returns:
            Model: Channel attention model
        """
        inputs = layers.Input(shape=(self.num_bands,))
        
        # Shared MLP
        x = layers.Dense(self.num_bands // self.reduction_ratio, activation='relu')(inputs)
        x = layers.Dense(self.num_bands, activation='sigmoid')(x)
        
        return Model(inputs, x)
    
    def _build_spatial_attention(self):
        """
        Build spatial attention branch
        
        Returns:
            Model: Spatial attention model
        """
        inputs = layers.Input(shape=(None, None, 1))
        
        # Convolutional layers
        x = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(inputs)
        
        return Model(inputs, x)
    
    def call(self, inputs):
        """
        Forward pass
        
        Args:
            inputs (tensor): Input tensor of shape (batch, height, width, channels)
            
        Returns:
            tensor: Output tensor with attention applied
        """
        # Channel attention
        channel_avg = tf.reduce_mean(inputs, axis=[1, 2])
        channel_weights = self.channel_attention(channel_avg)
        channel_weights = tf.expand_dims(tf.expand_dims(channel_weights, 1), 1)
        channel_out = inputs * channel_weights
        
        # Spatial attention
        spatial_avg = tf.reduce_mean(channel_out, axis=-1, keepdims=True)
        spatial_weights = self.spatial_attention(spatial_avg)
        spatial_out = channel_out * spatial_weights
        
        return spatial_out
    
    def get_config(self):
        """Get model configuration"""
        config = super(SpectralAttention, self).get_config()
        config.update({
            'num_bands': self.num_bands,
            'reduction_ratio': self.reduction_ratio
        })
        return config 