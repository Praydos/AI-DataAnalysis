"""
OpsGlean - Model stage 1: baseline Logistic Regression classifier.

Trains a class-weighted Logistic Regression on the bag-of-events features
(data/processed/train.csv) and evaluates it on the held-out test set.

Because the dataset is ~97% normal / ~3% anomalous, accuracy is not a
meaningful metric here -- we report precision, recall, F1, and the
confusion matrix instead.

Usage:
    python src/train_baseline.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_logreg.pkl"

EVENT_COLS = [f"E{i}" for i in range(1, 30)]  # E1..E29


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df[EVENT_COLS]
    y = df["y"]
    return X, y


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed train/test data not found. Run src/prepare_baseline_data.py first."
        )

    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    model = LogisticRegression(
        class_weight="balanced",  # compensates for the 2.93% anomaly rate
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Baseline Logistic Regression — Test Set Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion matrix (rows=actual, cols=predicted, [Normal, Anomaly]):")
    print(confusion_matrix(y_test, y_pred))

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
