import re
from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_data(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = ["tweet", "class"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[["tweet", "class"]].copy()
    df.rename(columns={"tweet": "text", "class": "label"}, inplace=True)

    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(clean_text)

    return df


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"@\w+", "", text)          # remove mentions
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)     # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text
