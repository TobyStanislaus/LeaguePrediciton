"""Quantify how much accuracy a leaky join buys, with a confidence interval.

Builds the same matches twice -- once joining the current ladder snapshot
regardless of time (``naive``, contaminated) and once undoing what happened
between kickoff and capture (``reconstructed``) -- then scores both on an
identical chronological split.

The two tables cover the same matches in the same order, so the comparison is
paired: the bootstrap resamples test matches and recomputes the difference on
each resample, which is what makes the interval meaningful on a few hundred
matches.

    python scripts/leakage_report.py --db data/cache/riot.sqlite --label apex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.cache import Cache  # noqa: E402
from features.build_features import LABEL_COLUMN, build_feature_table, feature_columns  # noqa: E402
from models.evaluate import assert_split_is_chronological, time_split  # noqa: E402
from models.train_baseline import build_model as build_linear  # noqa: E402
from models.train_boosted import build_model as build_boosted  # noqa: E402

MODELS = {"logistic": build_linear, "boosted": build_boosted}


def aligned_tables(db: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build naive and reconstructed tables restricted to their common matches."""
    with Cache(db) as cache:
        naive = build_feature_table(cache, mode="naive")
        reconstructed = build_feature_table(cache, mode="reconstructed")

    if naive.empty or reconstructed.empty:
        raise SystemExit("one of the modes produced no rows -- collect more data first")

    shared = sorted(set(naive["match_id"]) & set(reconstructed["match_id"]))
    naive = naive[naive["match_id"].isin(shared)].sort_values("game_start_ts")
    reconstructed = reconstructed[reconstructed["match_id"].isin(shared)].sort_values(
        "game_start_ts"
    )
    assert list(naive["match_id"]) == list(reconstructed["match_id"])
    return naive.reset_index(drop=True), reconstructed.reset_index(drop=True)


def predictions(table: pd.DataFrame, model_name: str, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    split = time_split(table, test_fraction=test_fraction)
    assert_split_is_chronological(split)
    model = MODELS[model_name]()
    model.fit(split.X_train, split.y_train)
    prob = model.predict_proba(split.X_test)[:, 1]
    return np.asarray(split.y_test, dtype=int), prob


def paired_bootstrap(
    y_true: np.ndarray,
    prob_leaky: np.ndarray,
    prob_clean: np.ndarray,
    iterations: int = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    """Bootstrap the accuracy and AUC gap, resampling test matches in pairs."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    acc_gaps = np.empty(iterations)
    auc_gaps = np.empty(iterations)

    leaky_hit = (prob_leaky >= 0.5).astype(int) == y_true
    clean_hit = (prob_clean >= 0.5).astype(int) == y_true

    for i in range(iterations):
        idx = rng.integers(0, n, n)
        acc_gaps[i] = leaky_hit[idx].mean() - clean_hit[idx].mean()
        sample_y = y_true[idx]
        if sample_y.min() == sample_y.max():
            auc_gaps[i] = np.nan
            continue
        auc_gaps[i] = roc_auc_score(sample_y, prob_leaky[idx]) - roc_auc_score(
            sample_y, prob_clean[idx]
        )

    return {
        "accuracy_gap": float(leaky_hit.mean() - clean_hit.mean()),
        "accuracy_lo": float(np.percentile(acc_gaps, 2.5)),
        "accuracy_hi": float(np.percentile(acc_gaps, 97.5)),
        "auc_gap": float(
            roc_auc_score(y_true, prob_leaky) - roc_auc_score(y_true, prob_clean)
        ),
        "auc_lo": float(np.nanpercentile(auc_gaps, 2.5)),
        "auc_hi": float(np.nanpercentile(auc_gaps, 97.5)),
        "share_positive": float((acc_gaps > 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/cache/riot.sqlite")
    parser.add_argument("--label", default="dataset")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    naive, reconstructed = aligned_tables(args.db)

    print(f"=== {args.label} ===")
    print(f"matches: {len(naive)}   features: {len(feature_columns())}")
    print(
        "time span: "
        f"{pd.to_datetime(naive['game_start_ts'].min(), unit='s')} -> "
        f"{pd.to_datetime(naive['game_start_ts'].max(), unit='s')}"
    )
    print(f"blue win rate: {naive[LABEL_COLUMN].mean():.3f}")

    rows = []
    for model_name in MODELS:
        y_true, prob_leaky = predictions(naive, model_name, args.test_fraction)
        _, prob_clean = predictions(reconstructed, model_name, args.test_fraction)
        stats = paired_bootstrap(y_true, prob_leaky, prob_clean, args.iterations)

        rows.append(
            {
                "model": model_name,
                "n_test": len(y_true),
                "acc_naive": round(float(((prob_leaky >= 0.5).astype(int) == y_true).mean()), 4),
                "acc_recon": round(float(((prob_clean >= 0.5).astype(int) == y_true).mean()), 4),
                "acc_gap": round(stats["accuracy_gap"], 4),
                "acc_95ci": f"[{stats['accuracy_lo']:+.3f}, {stats['accuracy_hi']:+.3f}]",
                "auc_gap": round(stats["auc_gap"], 4),
                "auc_95ci": f"[{stats['auc_lo']:+.3f}, {stats['auc_hi']:+.3f}]",
                "P(gap>0)": round(stats["share_positive"], 3),
            }
        )

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nA 95% interval spanning zero means this dataset is too small to resolve the leak,"
        "\nnot that the leak is absent -- the mechanism is proven separately by the unit tests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
