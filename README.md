# Hate Speech Classifier (Baseline, Naive Bayes, Linear SVM)

This graded project was carried out as part of the course Artificial Intelligence and Digital Skills in the AI & Society Minor at Leiden University. 

I implemented an end-to-end scikit-learn NLP pipeline (data cleaning, TF–IDF features, baseline/NB/SVM training, evaluation) and analyzed how class imbalance and metric choice can mask under-detection of hate speech, illustrating normative tradeoffs in moderation systems.

This project trains classical ML models to classify tweets into three categories:

- `0` = hate speech
- `1` = offensive language
- `2` = neither

Models implemented:

- Majority-class baseline (`DummyClassifier`)
- Multinomial Naive Bayes
- Linear SVM (`LinearSVC`)

## Repository Structure

- `src/data.py` - load + clean dataset
- `src/train.py` - train models and save artifacts
- `src/evaluate.py` - evaluate saved models
- `requirements.txt` - dependencies

## Data

This repository does not include the dataset file.

Expected input file:

- `data/labeled_data.csv`

Required columns in the CSV:

- `tweet` (text)
- `class` (label)

If you are using a Kaggle dataset, download it from Kaggle and place the CSV at `data/labeled_data.csv`.

## Setup

Create and activate a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

## Train

Trains three models and saves them as `.joblib` files in `artifacts/`:

```bash
python -m src.train --data_path data/labeled_data.csv --output_dir artifacts
```

Outputs:

- `artifacts/baseline.joblib`
- `artifacts/nb.joblib`
- `artifacts/svm.joblib`

Each artifact contains `(pipeline, X_test, y_test)`.

## Evaluate

Example:

```bash
python -m src.evaluate --model_path artifacts/svm.joblib
```

You can similarly evaluate:

- `artifacts/baseline.joblib`
- `artifacts/nb.joblib`

## Results (Test Split)

Note: accuracy is dominated by the majority class ("offensive language"), so macro-F1 and the hate-speech class metrics are more informative.

### Linear SVM (`class_weight="balanced"`)

- Accuracy: `0.89`
- Hate speech (class `0`):
- precision `0.40`
- recall `0.38`
- F1 `0.39`

Confusion matrix:

Hate speech is often misclassified as "offensive language" rather than "neither", suggesting systematic under-detection rather than random error.

### Naive Bayes

- Accuracy: `0.83`

Key pattern:

High precision but extremely low recall on hate speech (class `0`), meaning the classifier almost never flags hate speech.

This effectively prioritizes avoiding false accusations over harm detection.

(Full classification reports are available by running `src/evaluate.py`.)

## Interpretation (AI & Society)

Comparing a majority-class baseline, Naive Bayes, and a linear SVM illustrates that model choice encodes implicit normative priorities:

- Aggregate accuracy can look strong while performance on rare but socially critical categories (hate speech) remains weak.
- Naive Bayes exhibits near-zero hate-speech recall: a "low false positive" stance that can lead to under-enforcement.
- The SVM offers a better balance but still under-detects hate speech, raising concerns for real-world moderation settings where the cost of false negatives may be high.

This aligns with a common content moderation issue: standard metrics can obscure systematic failures on minority classes.

## Potential Improvements

- Thresholding / calibrated decision rules (especially if using models with decision scores or probabilities)
- Cost-sensitive learning / more systematic class-weight tuning
- Alternative models (logistic regression, character n-grams, or transformer baselines)
- Data quality: label ambiguity, annotator disagreement, and domain shift are likely limiting factors
