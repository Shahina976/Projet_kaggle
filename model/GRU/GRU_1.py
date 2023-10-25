from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

def create_model(input_shape):
    """
    Create a GRU-based model for time-series prediction.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: A GRU-based Keras model.
    """
    model = Sequential()

    # Ajout de couches GRU
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))

    # Couche de sortie
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # TimeDistributed car on a des séquences

    optimizer = Adam(learning_rate=0.00001)
    # Compilation du modèle
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
