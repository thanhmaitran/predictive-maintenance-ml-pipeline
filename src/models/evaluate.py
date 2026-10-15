"""
Model evaluation (Phases 11-12).

Because failures are rare (~3.39% positive rate), accuracy alone is
misleading. This module computes, per model:

    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

plus a confusion matrix, a threshold-sensitivity sweep (0.3-0.7), and
(for the tree-based model) feature importance.

Outputs:
    reports/metrics/model_comparison.csv
    reports/metrics/evaluation.json
    reports/figures/confusion_matrix.png
    reports/figures/roc_curve.png
    reports/figures/pr_curve.png
    reports/figures/feature_importance.png

Model selection criterion: primarily F1-score, with recall weighted
heavily due to the operational cost of a missed failure (false negative).
"""

import datetime
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.train import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN, RANDOM_STATE

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
PRIMARY_METRIC = "f1"


def _metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_models(test_path: str, models_dir: str, reports_dir: str) -> pd.DataFrame:
    """Evaluate all trained models on the held-out test set.

    Args:
        test_path: Path to data/processed/test.csv.
        models_dir: Directory containing fitted model artifacts.
        reports_dir: Directory to write metrics/figures to.

    Returns:
        DataFrame comparing models across all metrics.
    """
    df = pd.read_csv(test_path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Test data missing required columns: {missing}")

    X_test = df[FEATURE_COLUMNS]
    y_test = df[TARGET_COLUMN]

    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    (reports_path / "figures").mkdir(parents=True, exist_ok=True)
    (reports_path / "metrics").mkdir(parents=True, exist_ok=True)

    model_files = sorted(
        f for f in models_path.glob("*.joblib") if f.stem != "model_pipeline"
    )
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {models_dir}")

    comparison_rows = []
    evaluation_detail = {}

    for model_file in model_files:
        name = model_file.stem
        pipeline = joblib.load(model_file)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }
        comparison_rows.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        threshold_sweep = [_metrics_at_threshold(y_test, y_proba, t) for t in THRESHOLDS]

        evaluation_detail[name] = {
            "metrics": metrics,
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            },
            "threshold_sweep": threshold_sweep,
        }

        # Confusion matrix figure
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No Failure", "Failure"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}")
        fig.tight_layout()
        fig.savefig(reports_path / "figures" / f"confusion_matrix_{name}.png")
        plt.close(fig)

        # ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
        ax.set_title(f"ROC Curve — {name}")
        fig.tight_layout()
        fig.savefig(reports_path / "figures" / f"roc_curve_{name}.png")
        plt.close(fig)

        # PR curve
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
        ax.set_title(f"Precision-Recall Curve — {name}")
        fig.tight_layout()
        fig.savefig(reports_path / "figures" / f"pr_curve_{name}.png")
        plt.close(fig)

        # Threshold sweep plot
        sweep_df = pd.DataFrame(threshold_sweep)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", label="Precision")
        ax.plot(sweep_df["threshold"], sweep_df["recall"], marker="o", label="Recall")
        ax.plot(sweep_df["threshold"], sweep_df["f1"], marker="o", label="F1")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold sensitivity — {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_path / "figures" / f"threshold_sweep_{name}.png")
        plt.close(fig)

        # Feature importance (tree-based models only)
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            preprocessor = pipeline.named_steps["preprocessor"]
            cat_names = list(
                preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
            )
            all_feature_names = cat_names + NUMERIC_FEATURES
            importances = pd.Series(classifier.feature_importances_, index=all_feature_names)
            importances = importances.sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(importances.index, importances.values, color="#4C72B0")
            ax.set_title(f"Feature importance — {name}")
            fig.tight_layout()
            fig.savefig(reports_path / "figures" / f"feature_importance_{name}.png")
            plt.close(fig)

            evaluation_detail[name]["feature_importance"] = importances.sort_values(
                ascending=False
            ).round(4).to_dict()

    comparison_df = pd.DataFrame(comparison_rows).set_index("model")
    comparison_df.to_csv(reports_path / "metrics" / "model_comparison.csv")

    selected_model = comparison_df[PRIMARY_METRIC].idxmax()
    evaluation_detail["model_selection"] = {
        "primary_metric": PRIMARY_METRIC,
        "selected_model": selected_model,
    }

    with open(reports_path / "metrics" / "evaluation.json", "w") as f:
        json.dump(evaluation_detail, f, indent=2)

    print(comparison_df.round(4))
    print(f"\nSelected model (by {PRIMARY_METRIC}): {selected_model}")

    # --- Phase 13: save a standardized, self-contained artifact for the
    # selected model so src/models/predict.py has one canonical file to load.
    selected_pipeline = joblib.load(models_path / f"{selected_model}.joblib")
    joblib.dump(selected_pipeline, models_path / "model_pipeline.joblib")

    train_rows = None
    train_csv = Path("data/processed/train.csv")
    if train_csv.exists():
        train_rows = int(len(pd.read_csv(train_csv)))

    metadata = {
        "model": selected_model,
        "model_class": type(selected_pipeline.named_steps["classifier"]).__name__,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "dataset_version": "AI4I 2020 (ai4i2020.csv)",
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "training_rows": train_rows,
        "test_rows": int(len(df)),
        "metrics": evaluation_detail[selected_model]["metrics"],
        "selection_criterion": PRIMARY_METRIC,
    }
    with open(models_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Standardized artifact saved: {models_path / 'model_pipeline.joblib'}")
    print(f"Metadata saved: {models_path / 'model_metadata.json'}")

    return comparison_df


if __name__ == "__main__":
    evaluate_models(
        test_path="data/processed/test.csv",
        models_dir="models",
        reports_dir="reports",
    )
