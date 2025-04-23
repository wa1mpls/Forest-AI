import tensorflow as tf
from tensorflow.keras import layers, Model

class HybridForestModel(Model):
    def __init__(self, input_shape, gedi_features, num_classes):
        """
        Initialize hybrid forest model
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            gedi_features (int): Number of GEDI features
            num_classes (int): Number of output classes
        """
        super(HybridForestModel, self).__init__()
        
        # Image processing branch (CNN)
        self.image_branch = self._build_image_branch(input_shape)
        
        # GEDI features branch (MLP)
        self.gedi_branch = self._build_gedi_branch(gedi_features)
        
        # Combined processing
        self.combined_dense = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_classes)
        
    def _build_image_branch(self, input_shape):
        """
        Build CNN branch for image processing
        
        Args:
            input_shape (tuple): Shape of input images
            
        Returns:
            Model: CNN model
        """
        inputs = layers.Input(shape=input_shape)
        
        # Convolutional layers
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        return Model(inputs, x)
    
    def _build_gedi_branch(self, gedi_features):
        """
        Build MLP branch for GEDI features
        
        Args:
            gedi_features (int): Number of GEDI features
            
        Returns:
            Model: MLP model
        """
        inputs = layers.Input(shape=(gedi_features,))
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        return Model(inputs, x)
    
    def call(self, inputs):
        """
        Forward pass
        
        Args:
            inputs (tuple): (images, gedi_features)
            
        Returns:
            tensor: Model output
        """
        images, gedi_features = inputs
        
        # Process images
        image_features = self.image_branch(images)
        
        # Process GEDI features
        gedi_features = self.gedi_branch(gedi_features)
        
        # Combine features
        combined = tf.concat([image_features, gedi_features], axis=1)
        combined = self.combined_dense(combined)
        
        # Output layer
        output = self.output_layer(combined)
        
        return output
    
    def build_graph(self, image_shape, gedi_shape):
        """
        Build model graph for visualization
        
        Args:
            image_shape (tuple): Shape of input images
            gedi_shape (tuple): Shape of GEDI features
            
        Returns:
            Model: Model with specified input shapes
        """
        image_input = layers.Input(shape=image_shape)
        gedi_input = layers.Input(shape=gedi_shape)
        return Model(inputs=[image_input, gedi_input], outputs=self.call([image_input, gedi_input])) 