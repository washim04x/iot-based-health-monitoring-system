import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def save_data(train_data, test_data, path):
    """Save training and testing data to CSV files."""
    train_data.to_csv(os.path.join(path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(path, "test.csv"), index=False)

if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    print(home_dir)
    params_path = home_dir.as_posix() + "/params.yaml"
    
    params=yaml.safe_load(open(params_path))["make_dataset"]
    
    data_path = home_dir.as_posix() +"/data/raw/heart.csv"
    data = load_data(data_path)
    train_data, test_data = train_test_split(data, test_size=params["test_size"], random_state=params["random_state"])

    processed_data_path = home_dir.as_posix() + "/data/processed"
    os.makedirs(processed_data_path, exist_ok=True)

    save_data(train_data, test_data, processed_data_path)
