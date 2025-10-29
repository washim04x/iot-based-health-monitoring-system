import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import joblib
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer

def column_transformer(train_data):
    """Create a column transformer for feature engineering."""
    # label encoding for categorical features and scaling and normalization for numerical features
    # label encoding for categorical features and scaling and normalization for numerical features
    x_train= train_data.drop(columns=["HeartDisease"])
    y_train= train_data["HeartDisease"]
    trf = ColumnTransformer(
        [
            ("label_encoding", OrdinalEncoder(dtype="int64"), [1,2,6,8,10]), 
            ("scaling",        StandardScaler(),             [0,3,4,7]),
            ("normalization",  MinMaxScaler(),               [9]),
            ("keep_int",       "passthrough",                [5]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    train_transformed_data = trf.fit_transform(x_train)
    train_transformed_data["HeartDisease"] = y_train.values 


    return train_transformed_data, trf

def save_transformer(transformer, path):
    """Save the transformer to a file."""
    joblib.dump(transformer, path)

def save_transformed_data(data, path):
    """Save the transformed data to a CSV file."""
    data.to_csv(path, index=False)


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    processed_data_path = home_dir.as_posix() + "/data/processed/train.csv"

    train_data = pd.read_csv(processed_data_path)

    train_transformed_data, trf = column_transformer(train_data)

    # Save the transformer
    transformer_path = home_dir.as_posix() + "/models/build_features_transformer.joblib"
    save_transformer(trf, transformer_path)

    # Save the transformed training data
    transformed_data_path = home_dir.as_posix() + "/data/processed/train_transformed.csv"
    save_transformed_data(train_transformed_data, transformed_data_path)






