import pandas as pd
import argparse

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Clean and process event series data.")
    parser.add_argument("--input_csv", default="train_events.csv", help="Path to the input CSV file.")
    parser.add_argument("--input_parquet", default="train_series.parquet", help="Path to the input parquet file.")
    parser.add_argument("--output_csv", default="data_series_clean.csv", help="Path to the output CSV file.")
    
    return parser.parse_args()

def load_data():
    """
    Load the data from the given CSV and parquet files.
    
    Returns:
        DataFrame: data from train_events.csv.
        DataFrame: data from train_series.parquet.
    """
    data = pd.read_csv("train_events.csv")
    data_series = pd.read_parquet("train_series.parquet")
    
    return data, data_series

def clean_data(data):
    """
    Clean the data by removing rows with NaN values.
    
    Args:
        data (DataFrame): The original data.
        
    Returns:
        DataFrame: Cleaned data.
    """
    data_clean = data.dropna()
    data_clean = data_clean.copy()
    
    data_clean["step"] = data_clean["step"].astype("int")
    data_clean["event"] = data_clean["event"].replace({"onset": "sleeping", "wakeup": "awake"})
    
    return data_clean

def get_no_NaN_series(data):
    """
    Get series which don't have any NaN values.
    
    Args:
        data (DataFrame): The data with series_id and step columns.
    
    Returns:
        List: List of series_id which don't have any NaN values.
    """
    series_has_NaN = data.groupby('series_id')['step'].apply(lambda x: x.isnull().any())
    no_NaN_series = series_has_NaN[~series_has_NaN].index.tolist()
    
    # Remove these two "truncated" events series as seen in EDA
    no_NaN_series.remove('31011ade7c0a')
    no_NaN_series.remove('a596ad0b82aa')
    
    return no_NaN_series

def update_event_column(row, data_series_clean):
    """
    Set the event for a particular row in data_series_clean DataFrame.
    
    Args:
        row (Series): A row from the data_clean DataFrame.
        data_series_clean (DataFrame): Data to be updated.
    """
    mask = (
        (data_series_clean['series_id'] == row['series_id']) &
        (data_series_clean['step'] >= row['step']) &
        ((data_series_clean['step'] < row['step_next']) | (row['step_next'] == -1))
    )
    data_series_clean.loc[mask, 'event'] = row['event']

def main(args):
    # Load the data
    data, data_series = load_data(args.input_csv, args.input_parquet)
    
    # Clean the data
    data_clean = clean_data(data)
    
    # Get series without NaN values
    no_NaN_series = get_no_NaN_series(data)
    
    # Filter data_series based on no_NaN_series
    data_series_clean = data_series[data_series["series_id"].isin(no_NaN_series)]
    
    # Add 'event' column to data_series_clean
    data_series_clean['event'] = 'unknown'
    
    # Create ranges of steps
    data_clean['step_next'] = data_clean.groupby('series_id')['step'].shift(-1)
    data_clean['step_next'] = data_clean['step_next'].fillna(-1).astype(int)
    
    # Update 'event' column in data_series_clean
    data_clean.apply(update_event_column, axis=1, args=(data_series_clean,))
    
    # Replace 'unknown' events with 'awake'
    data_series_clean['event'] = data_series_clean['event'].replace('unknown', 'awake')
    
    # Save to CSV
    data_series_clean.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)