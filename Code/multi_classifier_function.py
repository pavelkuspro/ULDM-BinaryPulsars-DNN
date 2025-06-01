import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, BatchNormalization, MaxPooling1D, Flatten

def MultiClassifier(n):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(n, 1)))
    
    # First Convolutional Layer
    model.add(Conv1D(16, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Second Convolutional Layer
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Third Convolutional Layer
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten Layer
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2)) 

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4)) 

    # Output Layer for Binary Classification
    model.add(Dense(5, activation='softmax'))

    #optimizer = Adam(learning_rate=0.0005)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

