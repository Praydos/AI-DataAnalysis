"""
OpsGlean - shared model evaluation utilities.

Every model in this project (Logistic Regression baseline, future models
like Isolation Forest or the DeepLog-style LSTM) should be scored with
this same function, so comparisons between them are fair and consistent.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def evaluate(y_true, y_pred, y_proba=None, model_name="model", save_figures=True):
    """
    Compute and print standard binary classification metrics for anomaly
    detection, and (optionally) save a confusion matrix chart to
    reports/figures/. Returns a dict of metrics for logging/comparison.

    y_true, y_pred : array-like of 0/1 labels
    y_proba        : array-like of predicted probability of class 1 (optional,
                      needed for ROC-AUC)
    model_name     : label used in printed output and chart filenames
    save_figures   : if True, saves a confusion matrix PNG to reports/figures/
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {"precision": precision, "recall": recall, "f1": f1}

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    print(f"\n=== {model_name} — Test Set Results ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows=actual, cols=predicted, [Normal, Anomaly]):")
    print(cm)

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))

    if save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        slug = model_name.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{slug}_confusion_matrix.png", dpi=150)
        plt.close(fig)

        print(f"\nSaved confusion matrix chart to {FIGURES_DIR}/")

    return metrics