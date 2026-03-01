import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.logger import get_logger


logger = get_logger("data_ingestion")


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded successfully from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except KeyError as e:
        logger.error("Missing expected columns in the DataFrame: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred during data preprocessing: %s", e)
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug("Train and test data saved successfully at %s", raw_data_path)
    except Exception as e:
        logger.error("Error occurred while saving data: %s", e)
        raise


# main function to execute the data ingestion process
def main():
    try:
        test_size = 0.2
        data_path = (
            "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        )
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=2
        )
        save_data(train_data, test_data, data_path="./data")

    except Exception as e:
        logger.error("Failed to complete data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
