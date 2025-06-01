import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.models import Model
from tensorflow.keras import layers

class AnomalyDetector(Model):
    def __init__(self, n):
        super(AnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(n, 1)),

            layers.Conv1D(32, kernel_size=5, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.MaxPooling1D(pool_size=2, padding="same"),

            layers.Conv1D(16, kernel_size=5, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.MaxPooling1D(pool_size=2, padding="same"),

            layers.Conv1D(8, kernel_size=3, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.MaxPooling1D(pool_size=2, padding="same"),

            layers.Conv1D(4, kernel_size=3, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.MaxPooling1D(pool_size=2, padding="same"),
        ])
        
        # Decoder (mirror of the encoder)
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(4, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),

            layers.Conv1DTranspose(8, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),

            layers.Conv1DTranspose(16, kernel_size=5, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),

            layers.Conv1DTranspose(32, kernel_size=5, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.1),

            layers.Conv1D(1, kernel_size=3, activation="sigmoid", padding="same"),  # Output layer
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
