from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

def build_sequence_model(input_shape):
    """
    Construit un modèle de séquence basé sur LSTM pour la prédiction de séries temporelles.

    Args:
        input_shape (tuple): Forme des données d'entrée.

    Returns:
        Sequential: Un modèle Keras basé sur LSTM.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # TimeDistributed permet d'appliquer Dense à chaque étape de la séquence
    
    optimizer = Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # Ajustez en fonction de votre tâche
    
    return model
