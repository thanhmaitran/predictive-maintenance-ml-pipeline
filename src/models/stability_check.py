"""
Result-stability check (supplements Phase 11 evaluation).

The held-out test set contains only ~68 failures, so a single train/test
split gives noisy estimates: a handful of borderline cases flipping side
changes F1 by several points. In practice the Random Forest metrics
changed between two runs with identical code and random_state=42 simply
because the installed scikit-learn version differed.

This module quantifies that noise with repeated stratified k-fold
cross-validation on the full feature set (the same preprocessing + model
pipelines as src/models/train.py, refit inside every fold), and reports
mean +/- std for F1 / precision / recall / PR-AUC at threshold 0.5, plus
the mean F1 at each threshold in the 0.3-0.7 sweep.

Output: reports/metrics/stability_check.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.models.train import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, _build_pipelines

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def run_stability_check(
    features_path: str = "data/processed/features.csv",
    output_path: str = "reports/metrics/stability_check.json",
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict:
    df = pd.read_csv(features_path)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN].to_numpy()

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))

    results = {}
    for name in _build_pipelines():
        per_fold = {"f1": [], "precision": [], "recall": [], "pr_auc": []}
        f1_by_threshold = {t: [] for t in THRESHOLDS}
        for train_idx, test_idx in folds:
            pipeline = _build_pipelines()[name]  # fresh, unfitted pipeline per fold
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            per_fold["f1"].append(f1_score(y[test_idx], pred, zero_division=0))
            per_fold["precision"].append(precision_score(y[test_idx], pred, zero_division=0))
            per_fold["recall"].append(recall_score(y[test_idx], pred, zero_division=0))
            per_fold["pr_auc"].append(average_precision_score(y[test_idx], proba))
            for t in THRESHOLDS:
                f1_by_threshold[t].append(
                    f1_score(y[test_idx], (proba >= t).astype(int), zero_division=0)
                )

        results[name] = {
            "metrics_at_0.5": {
                m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                for m, v in per_fold.items()
            },
            "mean_f1_by_threshold": {str(t): float(np.mean(v)) for t, v in f1_by_threshold.items()},
        }

    report = {
        "method": f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats}, random_state={RANDOM_STATE})",
        "n_folds": len(folds),
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    for name, r in results.items():
        m = r["metrics_at_0.5"]
        print(
            f"{name}: F1 {m['f1']['mean']:.3f}±{m['f1']['std']:.3f} | "
            f"recall {m['recall']['mean']:.3f}±{m['recall']['std']:.3f} | "
            f"precision {m['precision']['mean']:.3f}±{m['precision']['std']:.3f} | "
            f"PR-AUC {m['pr_auc']['mean']:.3f}±{m['pr_auc']['std']:.3f}"
        )
        print("   mean F1 by threshold:", {k: round(v, 3) for k, v in r["mean_f1_by_threshold"].items()})
    return report


if __name__ == "__main__":
    run_stability_check()
