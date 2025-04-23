import tensorflow as tf
from tensorflow.keras import layers, Model

class EnhancedFeatures(Model):
    def __init__(self, num_bands):
        """
        Initialize enhanced features module
        
        Args:
            num_bands (int): Number of input spectral bands
        """
        super(EnhancedFeatures, self).__init__()
        
        self.num_bands = num_bands
        
        # Feature extraction layers
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        
        # Attention module
        self.attention = SpectralAttention(num_bands=256)
        
        # Feature fusion
        self.fusion = layers.Conv2D(128, 1, padding='same', activation='relu')
        
    def call(self, inputs):
        """
        Forward pass
        
        Args:
            inputs (tensor): Input tensor of shape (batch, height, width, channels)
            
        Returns:
            tensor: Enhanced features
        """
        # Feature extraction
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # Apply attention
        x3 = self.attention(x3)
        
        # Feature fusion
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.fusion(x)
        
        return x
    
    def get_config(self):
        """Get model configuration"""
        config = super(EnhancedFeatures, self).get_config()
        config.update({
            'num_bands': self.num_bands
        })
        return config 