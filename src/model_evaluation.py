import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import pickle
import json
from config.logger import get_logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

logger = get_logger("model_evaluation")


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Error loading the model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load test processed data"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Test data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading test data: %s", e)
        raise


def evaluate_model(clf, X_test: np.array, y_test: np.array) -> dict:
    """Evaluate model and return metrics"""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        matrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }
        logger.debug("Model evaluation completed with metrics: %s", matrics)

        return matrics
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save evaluation metrics to file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Evaluation metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving evaluation metrics: %s", e)
        raise


def main():
    try:
        clf = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
    except Exception as e:
        logger.error("Error in model evaluation pipeline: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
