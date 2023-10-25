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
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Assumer une sortie binaire
    
    optimizer = Adam(learning_rate=0.000001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
