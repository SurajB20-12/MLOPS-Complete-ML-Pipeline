import os
import string
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.logger import get_logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

logger = get_logger("data_preprocessing")


def transform_text(text):
    """
    Transforms the input text by performing the following steps:
    1. Converts the text to lowercase.
    2. Tokenizes the text into words.
    3. Removes non-alphanumeric characters.
    4. Removes stopwords and punctuation.
    5. Applies stemming to the remaining words.
    Args:
        text (str): The input text to be transformed.
    """
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]

    text = [
        word
        for word in text
        if word not in stopwords.words("english") and word not in string.punctuation
    ]

    text = [ps.stem(word) for word in text]
    return " ".join(text)


def preprocess_df(df, text_column="text", target_column="target"):
    """
    Preprocesses a DataFrame by applying text transformation and encoding target labels.

    """

    logger.debug("Starting DataFrame preprocessing")

    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])

    logger.debug("Target Column encoded successfully")

    df = df.drop_duplicates(keep="first")
    logger.debug("Duplicates removed successfully")

    df.loc[:, text_column] = df[text_column].apply(transform_text)
    logger.debug("Text transformation applied successfully")

    return df


def main(text_column="text", target_column="target"):
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded successfully")

        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed.csv"), index=False
        )
        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed.csv"), index=False
        )
        logger.debug("Preprocessed data saved successfully at %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Empty data error: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to complete data transformation process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
