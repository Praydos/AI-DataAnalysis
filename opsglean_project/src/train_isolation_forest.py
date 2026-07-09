"""
OpsGlean - Model stage 2: Isolation Forest (unsupervised baseline).

Unlike Logistic Regression, Isolation Forest does not use labels during
training -- it only learns what "normal" event-count patterns look like,
then flags points that are easy to isolate (few splits needed) as anomalies.

This matters as a real-world comparison: labeled anomaly data is often
scarce or unavailable in production, so an unsupervised method like this
is frequently the only option. We still use the labels here, but only to
evaluate performance after the fact -- not during training.

Usage:
    python src/train_isolation_forest.py
"""

import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluate import evaluate

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.pkl"

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

    # contamination = expected proportion of anomalies; we set it to match
    # the known ~2.93% rate from EDA, which is a realistic assumption an
    # ops team could set from historical incident rates even without full labels.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.0293,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)  # note: labels are NOT used here, unlike Logistic Regression

    # IsolationForest.predict returns 1 (normal) / -1 (anomaly); remap to 0/1
    raw_pred = model.predict(X_test)
    y_pred = (raw_pred == -1).astype(int)

    # score_samples: higher = more normal. Flip sign so higher = more anomalous,
    # matching the convention used for y_proba (probability of anomaly) in evaluate().
    y_score = -model.score_samples(X_test)

    evaluate(y_test, y_pred, y_score, model_name="Isolation Forest")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
