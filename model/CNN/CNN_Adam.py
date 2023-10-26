from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape):
    """
    Construit un modèle de séquence basé sur Conv1D pour la prédiction de séries temporelles.

    Args:
        input_shape (tuple): Forme des données d'entrée.

    Returns:
        Sequential: Un modèle Keras basé sur Conv1D.
    """
    model = Sequential()
    # 1D Convolutional layers
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Dense layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # TimeDistributed Output layer
    model.add(TimeDistributed(layers.Dense(WINDOW_SIZE, activation='sigmoid')))

    optimizer = Adam(learning_rate=0.000001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
