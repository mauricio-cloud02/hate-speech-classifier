import argparse
from pathlib import Path
from typing import Union

import joblib
from sklearn.metrics import classification_report, confusion_matrix


PathLike = Union[str, Path]


def evaluate(model_path: PathLike) -> None:
    model_path = Path(model_path)
    pipeline, X_test, y_test = joblib.load(model_path)
    y_pred = pipeline.predict(X_test)

    print("Model:", model_path)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model artifact and print standard classification metrics."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to a .joblib file produced by src.train (contains pipeline, X_test, y_test).",
    )
    args = parser.parse_args()
    evaluate(args.model_path)
