"""Performance monitoring charts: is the model limited by data, or by signal?

    python -m models.diagnostics --features data/processed/features.parquet

Produces ``artifacts/diagnostics.png``, four panels:

1. **Learning curve (accuracy)** -- validation accuracy against training-set
   size. The question that matters when a dataset is small: still climbing means
   collecting more matches pays off; flat means the signal is the ceiling and
   more of the same data will not help.
2. **Learning curve (log loss)** -- the same sweep scored on calibration rather
   than on thresholded decisions. Kept on its own axis rather than sharing one
   with accuracy: two measures on different scales in one frame is the single
   most misleading thing a chart can do.
3. **Training curve** -- validation log loss against boosting iterations, which
   is literally "accuracy as the model trains", and shows where it starts
   memorising instead of learning.
4. **Calibration** -- predicted probability against observed frequency on the
   held-out test set.

Colours are the Okabe-Ito set, which is designed to stay distinguishable under
the common forms of colour vision deficiency.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from features.build_features import LABEL_COLUMN, feature_columns
from models.evaluate import (
    DEFAULT_ARTIFACT_DIR,
    assert_3way_is_chronological,
    calibration_table,
    load_features,
    time_split_3way,
)

log = logging.getLogger(__name__)

# Okabe-Ito: a published colour-vision-deficiency-safe categorical set.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#999999"

INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------


def learning_curve(
    model_factory: Callable[[], Any],
    split,
    steps: int = 8,
    min_train: int = 40,
) -> pd.DataFrame:
    """Score validation performance as the training set grows.

    Training prefixes are taken in chronological order, never sampled: the
    model must always be trained on the past and scored on the future, at every
    size, or the curve measures something the deployed model never does.
    """
    n = len(split.X_train)
    if n < min_train:
        raise ValueError(f"need at least {min_train} training matches, have {n}")

    sizes = np.unique(np.linspace(min_train, n, steps, dtype=int))
    rows = []
    for size in sizes:
        X = split.X_train.iloc[:size]
        y = split.y_train.iloc[:size]
        if y.nunique() < 2:
            continue
        model = model_factory()
        model.fit(X, y)
        prob = model.predict_proba(split.X_val)[:, 1]
        rows.append(
            {
                "train_size": int(size),
                "val_accuracy": accuracy_score(split.y_val, (prob >= 0.5).astype(int)),
                "val_log_loss": log_loss(split.y_val, prob, labels=[0, 1]),
                "train_accuracy": accuracy_score(y, model.predict(X)),
            }
        )
    return pd.DataFrame(rows)


def boosting_curve(split, max_rounds: int = 400, step: int = 20) -> pd.DataFrame:
    """Validation log loss against the number of boosting rounds."""
    from models.train_boosted import build_model

    rows = []
    for n_estimators in range(step, max_rounds + 1, step):
        model = build_model(n_estimators=n_estimators)
        model.fit(split.X_train, split.y_train)
        prob = model.predict_proba(split.X_val)[:, 1]
        rows.append(
            {
                "n_estimators": n_estimators,
                "val_log_loss": log_loss(split.y_val, prob, labels=[0, 1]),
                "val_accuracy": accuracy_score(split.y_val, (prob >= 0.5).astype(int)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_diagnostics(
    curve: pd.DataFrame,
    boosting: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    out_path: Path,
    subtitle: str = "",
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Model diagnostics", fontsize=15, color=INK, x=0.02, ha="left", y=0.98)
    if subtitle:
        fig.text(0.02, 0.945, subtitle, fontsize=10, color=MUTED, ha="left")

    # 1 -- learning curve, accuracy
    ax = axes[0][0]
    ax.plot(curve["train_size"], curve["val_accuracy"], color=BLUE, linewidth=2,
            marker="o", markersize=5, label="validation")
    ax.plot(curve["train_size"], curve["train_accuracy"], color=ORANGE, linewidth=2,
            marker="o", markersize=5, linestyle="--", label="training")
    ax.axhline(0.5, color=GREY, linewidth=1, linestyle=":", label="coin flip")
    ax.set_title("Accuracy vs training size", fontsize=11, loc="left")
    ax.set_xlabel("training matches")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    _style(ax)

    # 2 -- learning curve, log loss (its own axis, deliberately not shared)
    ax = axes[0][1]
    ax.plot(curve["train_size"], curve["val_log_loss"], color=BLUE, linewidth=2,
            marker="o", markersize=5, label="validation")
    ax.axhline(np.log(2), color=GREY, linewidth=1, linestyle=":",
               label="always predict 0.5")
    ax.set_title("Log loss vs training size (lower is better)", fontsize=11, loc="left")
    ax.set_xlabel("training matches")
    ax.set_ylabel("log loss")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    _style(ax)

    # 3 -- training curve
    ax = axes[1][0]
    if boosting is not None and not boosting.empty:
        ax.plot(boosting["n_estimators"], boosting["val_log_loss"], color=GREEN,
                linewidth=2, marker="o", markersize=4, label="validation")
        ax.axhline(np.log(2), color=GREY, linewidth=1, linestyle=":",
                   label="always predict 0.5")
        best = boosting.loc[boosting["val_log_loss"].idxmin()]
        ax.axvline(best["n_estimators"], color=PURPLE, linewidth=1.5, linestyle="--",
                   label=f"best = {int(best['n_estimators'])}")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    else:
        ax.text(0.5, 0.5, "not enough data", ha="center", va="center", color=MUTED)
    ax.set_title("Log loss as boosting proceeds", fontsize=11, loc="left")
    ax.set_xlabel("boosting rounds")
    ax.set_ylabel("log loss")
    _style(ax)

    # 4 -- calibration
    ax = axes[1][1]
    if calibration is not None and not calibration.empty:
        ax.plot([0, 1], [0, 1], color=GREY, linewidth=1, linestyle=":",
                label="perfect calibration")
        ax.plot(calibration["predicted"], calibration["observed"], color=BLUE,
                linewidth=2, marker="o", markersize=6, label="model")
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "not enough data", ha="center", va="center", color=MUTED)
    ax.set_title("Calibration on held-out test matches", fontsize=11, loc="left")
    ax.set_xlabel("predicted probability of blue win")
    ax.set_ylabel("observed frequency")
    _style(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=400)
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--skip-boosting", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    from models.train_baseline import build_model as build_linear

    table = load_features(args.features)
    split = time_split_3way(table, args.val_fraction, args.test_fraction)
    assert_3way_is_chronological(split)
    log.info("train %d / validation %d / test %d", *split.sizes)

    curve = learning_curve(build_linear, split, steps=args.steps)
    print("\n=== learning curve (validation) ===")
    print(curve.round(4).to_string(index=False))

    if len(curve) >= 2:
        gain = curve["val_accuracy"].iloc[-1] - curve["val_accuracy"].iloc[0]
        print(
            f"\n  accuracy moved {gain:+.3f} between {curve['train_size'].iloc[0]} and "
            f"{curve['train_size'].iloc[-1]} training matches"
        )
        # Judge the second half of the curve against sampling noise, not a
        # fixed slope. A three-point fit on a 422-match validation set moves by
        # +/-0.02 on noise alone, so a bare slope threshold reports "still
        # climbing" for a curve that has visibly plateaued.
        half = len(curve) // 2
        gain = curve["val_accuracy"].iloc[-1] - curve["val_accuracy"].iloc[half]
        noise = float(np.sqrt(0.25 / max(len(split.y_val), 1)))
        print(f"  second half of the curve moved {gain:+.3f}; "
              f"noise floor on {len(split.y_val)} validation matches is +/-{noise:.3f}")

        if gain > 2 * noise:
            print("  still climbing -- more collection should pay off")
        elif gain > noise:
            print("  possibly still climbing, but within noise -- collect more to be sure")
        else:
            train_val_gap = (
                curve["train_accuracy"].iloc[-1] - curve["val_accuracy"].iloc[-1]
            )
            print("  plateaued -- more of the same data is unlikely to help.")
            if abs(train_val_gap) < 0.05:
                print(
                    f"  training and validation accuracy have converged "
                    f"(gap {train_val_gap:+.3f}), so this is a ceiling on what these "
                    "features can express, not overfitting. Better features beat more rows."
                )

    boosting = None
    if not args.skip_boosting:
        boosting = boosting_curve(split, max_rounds=args.max_rounds)
        best = boosting.loc[boosting["val_log_loss"].idxmin()]
        print(f"\n=== boosting curve ===\n  best validation log loss "
              f"{best['val_log_loss']:.4f} at {int(best['n_estimators'])} rounds")

    # Final calibration on the untouched test set.
    model = build_linear()
    model.fit(
        pd.concat([split.X_train, split.X_val]),
        pd.concat([split.y_train, split.y_val]),
    )
    test_prob = model.predict_proba(split.X_test)[:, 1]
    calibration = calibration_table(split.y_test, test_prob)
    print(f"\n=== held-out test ===\n  accuracy "
          f"{accuracy_score(split.y_test, (test_prob >= 0.5).astype(int)):.4f}"
          f" | log loss {log_loss(split.y_test, test_prob, labels=[0, 1]):.4f}"
          f" | n={len(split.y_test)}")

    written = plot_diagnostics(
        curve, boosting, calibration,
        Path(args.artifacts) / "diagnostics.png",
        subtitle=f"{len(table)} matches | train {split.sizes[0]} / "
                 f"validation {split.sizes[1]} / test {split.sizes[2]}",
    )
    print(f"\n  charts -> {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
