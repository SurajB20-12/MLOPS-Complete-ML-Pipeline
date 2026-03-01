import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import pickle
from config.logger import get_logger
from sklearn.ensemble import RandomForestClassifier

logger = get_logger("model_training")


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def train_model(
    X_train: np.array, y_train: np.array, params: dict
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier model.
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        params (dict): Hyperparameters for the Random Forest model.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "Number of samples in X_train and y_train must be the same."
            )

        logger.debug("Initialize Random Forest Classifier with parameters: %s", params)
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"], random_state=params["random_state"]
        )
        logger.debug("Model training started with %d samples", X_train.shape[0])

        clf.fit(X_train, y_train)
        logger.debug("Model training completed")
        return clf
    except ValueError as e:
        logger.error("Value error during model training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", file_path)
    except FileNotFoundError as e:
        logger.error("Directory not found for saving the model: %s", e)
        raise
    except Exception as e:
        logger.error("Error occurred while saving the model: %s", e)
        raise


def main():
    try:
        params = {"n_estimators": 25, "random_state": 2}

        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        model_save_path = "models/model.pkl"
        save_model(clf, model_save_path)
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
