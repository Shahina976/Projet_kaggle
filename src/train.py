import argparse
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

# Importez la fonction qui crée le modèle depuis le script approprié dans le répertoire '../model'.
from model.your_model_script import create_model
from data_split import read_file, data_split

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model on processed data.")
    parser.add_argument("--input_csv", default="data_series_clean.csv", help="Path to the input CSV file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    
    return parser.parse_args()

def preprocess_data(data):
    """
    Preprocess the data before feeding it into the model.
    
    Args:
        data (DataFrame): Input data.
    
    Returns:
        DataFrame: Preprocessed data.
    """
    # Placeholder function for preprocessing; implement your own steps.
    return data

def main():
    """
    Main function to execute the data processing and model training process.
    """
    args = parse_args()

    data = read_file(args.input_csv)
    data_train, data_test = data_split(data)

    X_train, y_train = preprocess_data(data_train), data_train["event"]
    X_test, y_test = preprocess_data(data_test), data_test["event"]

    model = create_model(X_train.shape[1])
    early_stop = EarlyStopping(patience=args.patience)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=args.batch_size, epochs=args.epochs, callbacks=[early_stop])

    # Save the model
    model.save("saved_model_directory")

if __name__ == "__main__":
    main()
