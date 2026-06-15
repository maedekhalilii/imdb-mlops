import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Make sure NLTK resources exist
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def clean_text(text: str) -> str:
    """
    Cleans a single review text:
    - Lowercase
    - Remove HTML tags
    - Remove non-alphanumeric characters
    - Remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning + sentiment mapping.
    """
    df = df.copy()
    df["review"] = df["review"].apply(clean_text)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df

def tokenize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenizes text after preprocessing.
    """
    df = df.copy()
    df["tokens"] = df["review"].apply(word_tokenize)
    return df
