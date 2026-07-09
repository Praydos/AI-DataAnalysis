"""
OpsGlean - Data pipeline stage 1: prepare baseline features.

Loads data/raw/Event_occurrence_matrix.csv (bag-of-events counts per block)
and produces a stratified train/test split that preserves the dataset's
~2.93% anomaly rate in both sets. Writes results to data/processed/ so
every downstream step (model training, evaluation) uses an identical split.

Usage:
    python src/prepare_baseline_data.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Event_occurrence_matrix.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EVENT_COLS = [f"E{i}" for i in range(1, 30)]  # E1..E29
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["y"] = (df["Label"] == "Fail").astype(int)  # 1 = anomaly, 0 = normal
    return df


def split(df: pd.DataFrame):
    X = df[EVENT_COLS]
    y = df["y"]
    block_ids = df["BlockId"]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, block_ids,
        test_size=TEST_SIZE,
        stratify=y,  # preserves the anomaly ratio in both splits
        random_state=RANDOM_STATE,
    )
    train_df = X_train.assign(BlockId=id_train.values, y=y_train.values)
    test_df = X_test.assign(BlockId=id_test.values, y=y_test.values)
    return train_df, test_df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. See docs/DATA.md for how to obtain the raw data."
        )

    df = load_features(RAW_PATH)
    train_df, test_df = split(df)

    print(f"Train set: {len(train_df):,} rows, {train_df['y'].sum():,} anomalies "
          f"({100 * train_df['y'].mean():.2f}%)")
    print(f"Test set:  {len(test_df):,} rows, {test_df['y'].sum():,} anomalies "
          f"({100 * test_df['y'].mean():.2f}%)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\nSaved to {PROCESSED_DIR}/train.csv and {PROCESSED_DIR}/test.csv")


if __name__ == "__main__":
    main()
