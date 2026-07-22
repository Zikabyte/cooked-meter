import os

import pandas as pd
import re
from datasets import load_dataset

# ------------------------------ Dataset Config ------------------------------ #
DATA_SET_2_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned", "PRDECT-ID_cleaned.csv")
DATA_SET_3_HF = "AIbnuHibban/e-commerce-sentiment-bahasa-indonesia" # Hugging Face Path

# ---------------------------------- Cleaners --------------------------------- #
def tokenize(text):
    """Lowercase a review and split it into alphabetic word tokens."""
    return re.findall(r"\b[a-z]+\b", str(text).lower())

def clean_review_text(text):
    """Replicate the Review_Clean preprocessing (lowercase, strip URLs,
    collapse excessively repeated letters/punctuation, normalize whitespace)
    on new, raw review text -- e.g. a review typed live in demo.ipynb --
    so it matches what Model A/B were trained on."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_dataset3():
    """Load Dataset 3 from Hugging Face and clean it: flag rows with a
    linguistic-phenomenon category, drop full-row duplicates, and add
    tokenized/comment_length helper columns."""
    df_raw = load_dataset(DATA_SET_3_HF)["train"].to_pandas()
    df_cleaned = df_raw.copy()

    df_cleaned["category"] = df_cleaned["category"].fillna("tidak_berkategori")
    df_cleaned["has_linguistic_label"] = df_cleaned["category"] != "tidak_berkategori"

    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)

    df_cleaned["comment"] = df_cleaned["comment"].str.strip()
    df_cleaned["tokens"] = df_cleaned["comment"].apply(tokenize)
    df_cleaned["comment_length"] = df_cleaned["tokens"].apply(len)

    return df_cleaned

# ---------------------------------- Loader ---------------------------------- #
def load_datasets():
    """Load the cleaned Dataset 2 (PRDECT-ID) CSV and the freshly-cleaned
    Dataset 3 (Hugging Face), returned as (df2, df3)."""
    df2 = pd.read_csv(DATA_SET_2_PATH)
    df3 = clean_dataset3()

    return (df2, df3)

# ---------------------------------- Preview --------------------------------- #
def main():
    """Load both datasets and print a quick .info() summary of each."""
    df = load_datasets()

    print(df[0].info())
    print(df[1].info())

if __name__ == '__main__':
    main()