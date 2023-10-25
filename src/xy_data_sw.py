import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(data):
    """
    Preprocesses the data by scaling and encoding relevant features.

    Args:
        data (pd.DataFrame): Raw data to preprocess.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    scaler = MinMaxScaler()
    data[['anglez', 'enmo']] = scaler.fit_transform(data[['anglez', 'enmo']])
    label_encoder = LabelEncoder()
    data['event'] = label_encoder.fit_transform(data['event'])
    return data

def create_dataset(data, window_size):
    """
    Transforms the data into a windowed dataset suitable for time-series prediction.

    Args:
        data (pd.DataFrame): Preprocessed data.
        window_size (int): Desired window size for the dataset.

    Returns:
        np.ndarray, np.ndarray: x (features) and y (labels) datasets.
    """
    grouped = data.groupby("series_id")

    total_windows = sum(len(group) - window_size + 1 for _, group in grouped)

    x = np.empty((total_windows, window_size, 2))
    y = np.empty((total_windows, window_size))

    start_idx = 0
    for _, group in grouped:
        group_values = group[['anglez', 'enmo']].values
        group_labels = group['event'].values
        for i in range(len(group) - window_size + 1):
            x[start_idx] = group_values[i:i+window_size]
            y[start_idx] = group_labels[i:i+window_size]
            start_idx += 1

    return x, y
