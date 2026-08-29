"""Logistic regression baseline on the pre-game team feature table.

Deliberately simple and the first thing to run: a linear model on rank/LP/
winrate differences is close to the honest ceiling for this problem, and it is
the reference every fancier model has to beat.

    python -m models.train_baseline --features data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.build_features import feature_columns
from models.evaluate import DEFAULT_ARTIFACT_DIR, load_features, run_experiment

log = logging.getLogger(__name__)


def build_model(C: float | None = None, cv_folds: int = 4) -> Pipeline:
    """Impute, scale, then fit a regularised linear model.

    Imputation matters because a player occasionally has no ranked entry;
    scaling matters because LP and winrate live on wildly different scales.

    Regularisation strength is chosen by cross-validation rather than fixed.
    With a few hundred matches and 27 correlated features, an under-regularised
    fit overfits badly enough to score worse than a coin flip on log loss. The
    folds are time-ordered (``TimeSeriesSplit``) so tuning never peeks forward,
    which a plain k-fold would. Pass ``C`` to pin it instead.
    """
    if C is not None:
        classifier = LogisticRegression(C=C, max_iter=5000)
    else:
        classifier = GridSearchCV(
            LogisticRegression(max_iter=5000),
            {"C": np.logspace(-4, 2, 13)},
            cv=TimeSeriesSplit(n_splits=cv_folds),
            scoring="neg_log_loss",
        )

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", classifier),
        ]
    )


def show_coefficients(model: Pipeline, top: int = 10) -> None:
    """Print the strongest weights -- a quick smell test for leakage."""
    classifier = model.named_steps["clf"]
    if hasattr(classifier, "best_estimator_"):
        print(f"\n  cross-validated C = {classifier.best_params_['C']:.4g}")
        classifier = classifier.best_estimator_

    weights = pd.Series(classifier.coef_[0], index=feature_columns())
    ranked = weights.reindex(weights.abs().sort_values(ascending=False).index)

    print("\n  strongest coefficients (standardised)")
    for name, value in ranked.head(top).items():
        print(f"    {value:+.4f}  {name}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--C", type=float, default=None,
        help="pin inverse regularisation strength; default is time-series cross-validated",
    )
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    table = load_features(args.features)
    plot = Path(args.artifacts) / "calibration_baseline.png"

    model = build_model(C=args.C)
    metrics, _, _ = run_experiment(
        "logistic regression", model, table, args.test_fraction, plot
    )
    show_coefficients(model)

    return 1 if metrics.accuracy >= 0.80 else 0


if __name__ == "__main__":
    raise SystemExit(main())
