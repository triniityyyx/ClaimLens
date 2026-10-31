"""
ClaimLens — Machine Learning Training Pipeline
Trains a Scikit-learn Pipeline (TF-IDF Vectorizer + Logistic Regression)
on real labeled news data with strict train/test split hygiene,
metric evaluation, feature coefficient analysis, and Joblib artifact serialization.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "train_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "claimlens_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def load_and_preprocess_dataset(
    data_path: str = DEFAULT_DATA_PATH,
    true_csv: str = None,
    fake_csv: str = None,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Load dataset from either a single CSV file or separate True/Fake CSV files.
    Cleans missing values, normalizes labels to 1 (Real) and 0 (Fake),
    and deduplicates records.
    """
    if true_csv and fake_csv:
        if not os.path.exists(true_csv) or not os.path.exists(fake_csv):
            raise FileNotFoundError(
                f"Specified dataset files not found: {true_csv} or {fake_csv}"
            )
        print(f"[DATA] Loading separate datasets:\n  True: {true_csv}\n  Fake: {fake_csv}")
        df_true = pd.read_csv(true_csv)
        df_fake = pd.read_csv(fake_csv)

        # Standardize text column
        t_col = text_col if text_col in df_true.columns else df_true.columns[0]
        f_col = text_col if text_col in df_fake.columns else df_fake.columns[0]

        df_true = df_true[[t_col]].rename(columns={t_col: "text"})
        df_true["label"] = 1

        df_fake = df_fake[[f_col]].rename(columns={f_col: "text"})
        df_fake["label"] = 0

        df = pd.concat([df_true, df_fake], ignore_index=True)
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Training dataset not found at: {data_path}\n"
                "Please place your dataset in the data/ folder or specify --data <path>."
            )
        print(f"[DATA] Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

        if text_col not in df.columns or label_col not in df.columns:
            raise KeyError(
                f"Columns '{text_col}' or '{label_col}' not found. Available: {list(df.columns)}"
            )
        df = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )

    # Clean text records
    df["text"] = df["text"].astype(str).apply(clean_text)
    # Drop empty texts
    df = df[df["text"].str.len() > 10].copy()

    # Normalize labels: map string or numeric labels to 1 (Real) and 0 (Fake)
    def normalize_label(val):
        s = str(val).strip().lower()
        if s in ["1", "real", "true", "factual", "credible"]:
            return 1
        elif s in ["0", "fake", "false", "misleading", "hoax"]:
            return 0
        try:
            return 1 if int(val) == 1 else 0
        except Exception:
            return 0

    df["label"] = df["label"].apply(normalize_label)

    # Deduplicate by text to prevent identical articles across train and test sets
    initial_count = len(df)
    df = df.drop_duplicates(subset=["text"])
    dedup_count = len(df)
    if initial_count != dedup_count:
        print(f"[DATA] Deduplicated {initial_count - dedup_count} identical records.")

    print(
        f"[DATA] Preprocessed {len(df)} samples: "
        f"Real (1) = {(df['label'] == 1).sum()}, Fake (0) = {(df['label'] == 0).sum()}"
    )
    return df


def train_pipeline(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> Tuple[Pipeline, Dict]:
    """
    Split dataset, fit TF-IDF and Logistic Regression on training set only,
    evaluate on held-out test set, and compute explainability features.
    """
    X = np.array(df["text"].tolist(), dtype=object)
    y = np.array(df["label"].tolist(), dtype=int)

    # Stratified split to preserve class balance in both subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[TRAIN] Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Construct Scikit-learn Pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10000,
                    stop_words="english",
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Fit exclusively on training data (strictly preventing train/test leakage)
    print("[TRAIN] Fitting TF-IDF Vectorizer and Logistic Regression...")
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 50)
    print(" CLAIMLENS MODEL EVALUATION (HELD-OUT TEST SET)")
    print("=" * 50)
    print(f" Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Precision: {prec:.4f}")
    print(f" Recall   : {rec:.4f}")
    print(f" F1-Score : {f1:.4f}")
    print(f" Confusion Matrix (TN, FP / FN, TP):\n  {cm}")
    print("=" * 50 + "\n")

    # Extract top indicative features from model coefficients
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    # Positive coefficients favor class 1 (Real), negative favor class 0 (Fake)
    top_real_idx = np.argsort(coefs)[-10:][::-1]
    top_fake_idx = np.argsort(coefs)[:10]

    top_real_words = [
        {"term": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
        for i in top_real_idx
    ]
    top_fake_words = [
        {"term": str(feature_names[i]), "weight": round(float(abs(coefs[i])), 4)}
        for i in top_fake_idx
    ]

    metadata = {
        "model_type": "Logistic Regression with TF-IDF Vectorizer",
        "training_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        },
        "top_features": {
            "real_indicators": top_real_words,
            "fake_indicators": top_fake_words,
        },
        "vocabulary_size": len(feature_names),
        "label_mapping": {"1": "LIKELY REAL", "0": "LIKELY FAKE"},
    }

    return pipeline, metadata


def save_artifacts(
    pipeline: Pipeline,
    metadata: Dict,
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
):
    """Serialize model pipeline with Joblib and save metadata JSON."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[SAVE] Trained model pipeline saved to: {model_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVE] Model metadata & evaluation metrics saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ClaimLens — Train TF-IDF + Logistic Regression Fake News Classifier"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to single CSV dataset file",
    )
    parser.add_argument(
        "--true-csv",
        type=str,
        default=None,
        help="Path to True/Real news CSV (e.g. ISOT dataset True.csv)",
    )
    parser.add_argument(
        "--fake-csv",
        type=str,
        default=None,
        help="Path to Fake news CSV (e.g. ISOT dataset Fake.csv)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Column name containing news text",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Column name containing binary label (1=Real, 0=Fake)",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Destination path for .joblib model",
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help="Destination path for metadata JSON",
    )

    args = parser.parse_args()

    df = load_and_preprocess_dataset(
        data_path=args.data,
        true_csv=args.true_csv,
        fake_csv=args.fake_csv,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    pipeline, metadata = train_pipeline(df)
    save_artifacts(pipeline, metadata, args.model_out, args.meta_out)
    print("\n[SUCCESS] Model training and evaluation successfully completed!")


if __name__ == "__main__":
    main()
