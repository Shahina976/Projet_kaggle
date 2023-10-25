import pandas as pd
import argparse

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Read and split data from a CSV file into train and test datasets.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    
    return parser.parse_args()

def read_file(file_path):
    """
    Read data from the specified CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        DataFrame: Data from the specified CSV file.
    """
    return pd.read_csv(file_path)

def data_split(data):
    """
    Split the data into train and test datasets based on unique series_id values.
    
    Args:
        data (DataFrame): The data to be split.
        
    Returns:
        DataFrame, DataFrame: train and test datasets.
    """
    # Recover uniques series_id of the dataset
    series_id_list = list(data.series_id.unique())
    
    # Select 25 first individuals for data_train
    data_train = data[data["series_id"].isin(series_id_list[:25])]

    # Select the rest for data_test
    data_test = data[data["series_id"].isin(series_id_list[25:])]
    
    return data_train, data_test

def main():
    args = parse_args()
    
    # Read data from the CSV file
    data = read_file(args.input_csv)
    
    # Split data into train and test datasets
    data_train, data_test = data_split(data)
    
    return data_train, data_test

if __name__ == "__main__":
    main()
