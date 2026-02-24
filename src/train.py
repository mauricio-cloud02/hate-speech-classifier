import argparse
from pathlib import Path
from typing import Union

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

from src.data import load_data


PathLike = Union[str, Path]


def train(data_path: PathLike, output_dir: PathLike = "artifacts") -> None:
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    df = load_data(data_path)

    X = df["text"]
    y = df["label"]

    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset size: {len(df)} rows")
    print("Class distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir.resolve()}")

    models = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "svm": LinearSVC(class_weight="balanced"),
        "nb": MultinomialNB(),
    }

    for name, model in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2)
            )),
            ("clf", model),
        ])

        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        model_path = output_dir / f"{name}.joblib"
        joblib.dump(
            (pipeline, X_test, y_test),
            model_path
        )
        print(f"Saved: {model_path}")

    print("Training complete.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline, Naive Bayes, and linear SVM text classifiers."
    )
    parser.add_argument(
        "--data_path",
        default="data/labeled_data.csv",
        help="Path to the input CSV file (must include 'tweet' and 'class' columns).",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts",
        help="Directory where trained .joblib model artifacts will be saved.",
    )

    args = parser.parse_args()

    train(args.data_path, args.output_dir)
