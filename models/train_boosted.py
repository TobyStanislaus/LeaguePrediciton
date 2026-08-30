"""Gradient-boosted model (LightGBM, falling back to XGBoost) on the same features.

Scored identically to the baseline so the comparison is fair. On rank/LP-only
features a boosted model usually gains very little over logistic regression --
the relationship really is close to linear in rank difference. A large gap is
more likely to indicate leakage or overfitting than a genuine discovery.

    python -m models.train_boosted --features data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from features.build_features import feature_columns
from models.evaluate import DEFAULT_ARTIFACT_DIR, load_features, run_experiment, save_model

log = logging.getLogger(__name__)


def build_model(n_estimators: int = 200, learning_rate: float = 0.03, max_depth: int = 3) -> Any:
    """LightGBM if available, else XGBoost. Both handle NaN natively.

    Deliberately small trees, heavy sub-sampling and strong L2. With a few
    hundred matches and a weak signal, a boosted model will happily memorise
    noise and end up worse than a coin flip on log loss. These defaults are
    tuned for that regime and should be loosened once the dataset grows.
    """
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=7,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            verbose=-1,
        )
    except ImportError:
        log.info("lightgbm unavailable, using xgboost")
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            eval_metric="logloss",
            tree_method="hist",
        )


def show_importances(model: Any, top: int = 10) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    ranked = pd.Series(importances, index=feature_columns()).sort_values(ascending=False)
    print("\n  strongest feature importances")
    for name, value in ranked.head(top).items():
        print(f"    {value:>10.4f}  {name}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument(
        "--save", default="artifacts/model_boosted.joblib",
        help="where to persist the fitted model; pass an empty string to skip",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    table = load_features(args.features)
    plot = Path(args.artifacts) / "calibration_boosted.png"

    model = build_model(args.n_estimators, args.learning_rate, args.max_depth)
    metrics, _, _ = run_experiment("gradient boosting", model, table, args.test_fraction, plot)
    show_importances(model)

    if args.save:
        saved = save_model(
            model,
            args.save,
            {
                "name": "gradient boosting",
                "mode": str(table["leakage_mode"].iloc[0]),
                "n_train": metrics.n_train,
                "metrics": metrics.as_dict(),
            },
        )
        print(f"\n  model saved -> {saved}")

    return 1 if metrics.accuracy >= 0.80 else 0


if __name__ == "__main__":
    raise SystemExit(main())
