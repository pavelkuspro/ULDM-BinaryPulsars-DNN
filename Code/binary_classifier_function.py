import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, BatchNormalization, MaxPooling1D, Flatten

def BinaryClassifier(n):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(n, 1)))
    
    # First Convolutional Layer
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())  # Helps stabilize training
    model.add(MaxPooling1D(pool_size=2))

    # Second Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Flatten Layer
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))  # Reduces overfitting

    # Output Layer for Binary Classification
    model.add(Dense(1, activation='sigmoid'))

    return model
