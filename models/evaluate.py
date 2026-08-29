"""Time-based splitting, metrics, calibration, and the leakage sanity check.

Shared by both training scripts so the baseline and the boosted model are
scored identically.

Why a time split rather than a shuffle
--------------------------------------
Rank and LP distributions drift: ladders inflate through a season, players climb
and demote, and Riot's rank distribution shifts after each soft reset. A random
shuffle lets the model see the future and interpolate, which overstates
accuracy. Training on earlier matches and testing on later ones measures what
this model would actually do in use.

Why calibration, not just accuracy
----------------------------------
For a coin-flip-ish problem, a model that says "58% blue" and is right 58% of
the time is far more useful than one that is confidently wrong at the same
accuracy. Accuracy alone hides that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from features.build_features import LABEL_COLUMN, feature_columns

log = logging.getLogger(__name__)

# Rank/LP alone carries limited signal. Published work and the structure of
# matchmaking put an honest model in roughly this band; matchmaking is designed
# to produce even games, which caps how well pre-game rank can predict.
EXPECTED_ACCURACY = (0.55, 0.65)
SUSPICIOUS_ACCURACY = 0.70
ALARMING_ACCURACY = 0.80

DEFAULT_ARTIFACT_DIR = Path("artifacts")


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    cutoff_ts: float
    # Kickoff times are carried alongside so the chronology guard can inspect
    # what actually ended up on each side, rather than trusting row counts.
    train_ts: pd.Series
    test_ts: pd.Series

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.X_train), len(self.X_test)


def time_split(table: pd.DataFrame, test_fraction: float = 0.25) -> Split:
    """Split chronologically: earliest matches train, latest matches test."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if table.empty:
        raise ValueError("cannot split an empty feature table")

    ordered = table.sort_values("game_start_ts").reset_index(drop=True)
    cutoff_index = int(len(ordered) * (1.0 - test_fraction))
    cutoff_index = max(1, min(cutoff_index, len(ordered) - 1))

    train = ordered.iloc[:cutoff_index]
    test = ordered.iloc[cutoff_index:]

    columns = feature_columns()
    return Split(
        X_train=train[columns],
        X_test=test[columns],
        y_train=train[LABEL_COLUMN],
        y_test=test[LABEL_COLUMN],
        cutoff_ts=float(test["game_start_ts"].iloc[0]),
        train_ts=train["game_start_ts"],
        test_ts=test["game_start_ts"],
    )


def assert_split_is_chronological(split: Split) -> None:
    """Guard against a shuffle sneaking back in.

    Checks the kickoff times actually present on each side. An earlier version
    compared row counts against a re-sorted table, which was true by
    construction and would not have caught anything.
    """
    if len(split.train_ts) == 0 or len(split.test_ts) == 0:
        raise AssertionError("one side of the split is empty")

    train_max = float(split.train_ts.max())
    test_min = float(split.test_ts.min())
    if train_max > test_min:
        raise AssertionError(
            "training data contains matches later than the test set "
            f"(train max {train_max:.0f} > test min {test_min:.0f})"
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass
class Metrics:
    name: str
    accuracy: float
    log_loss: float
    brier: float
    roc_auc: float
    base_rate: float
    majority_accuracy: float
    n_train: int
    n_test: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "accuracy": round(self.accuracy, 4),
            "log_loss": round(self.log_loss, 4),
            "brier": round(self.brier, 4),
            "roc_auc": round(self.roc_auc, 4),
            "base_rate": round(self.base_rate, 4),
            "majority_accuracy": round(self.majority_accuracy, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }

    def render(self) -> str:
        lines = [
            f"  accuracy          {self.accuracy:.4f}",
            f"  log loss          {self.log_loss:.4f}   (coin flip = {np.log(2):.4f})",
            f"  Brier score       {self.brier:.4f}   (coin flip = 0.2500)",
            f"  ROC AUC           {self.roc_auc:.4f}",
            f"  always-blue       {self.majority_accuracy:.4f}   (beat this to be useful)",
            f"  train / test      {self.n_train} / {self.n_test}",
        ]
        return "\n".join(lines)


def evaluate_predictions(
    name: str, y_true: pd.Series, y_prob: np.ndarray, n_train: int
) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1 - 1e-9)
    y_pred = (y_prob >= 0.5).astype(int)

    base_rate = float(y_true.mean())
    metrics = Metrics(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        base_rate=base_rate,
        majority_accuracy=float(max(base_rate, 1.0 - base_rate)),
        n_train=n_train,
        n_test=len(y_true),
    )
    metrics.warnings = leakage_warnings(metrics)
    return metrics


def leakage_warnings(metrics: Metrics) -> list[str]:
    """Flag results too good to be true for rank/LP-only features."""
    warnings: list[str] = []
    low, high = EXPECTED_ACCURACY

    if metrics.accuracy >= ALARMING_ACCURACY:
        warnings.append(
            f"accuracy {metrics.accuracy:.1%} is far above the {low:.0%}-{high:.0%} band "
            "plausible from rank/LP alone. Treat this as a leakage bug until proven "
            "otherwise -- check the feature/label join and the snapshot timing."
        )
    elif metrics.accuracy >= SUSPICIOUS_ACCURACY:
        warnings.append(
            f"accuracy {metrics.accuracy:.1%} is above the expected {low:.0%}-{high:.0%} band. "
            "Worth investigating before believing it."
        )
    elif metrics.accuracy < 0.50:
        warnings.append(
            "accuracy is below chance -- likely too little data, or an inverted label."
        )
    elif metrics.accuracy < low:
        warnings.append(
            f"accuracy {metrics.accuracy:.1%} is below the expected {low:.0%}-{high:.0%} band. "
            "Not a leakage problem -- usually too few matches, or a sample where matchmaking "
            "has already equalised the ranks (apex tiers especially)."
        )

    if metrics.log_loss > np.log(2):
        warnings.append(
            f"log loss {metrics.log_loss:.4f} is worse than always predicting 0.5 "
            f"({np.log(2):.4f}). The model is overconfident -- regularise harder or "
            "collect more matches."
        )

    if metrics.n_test < 100:
        warnings.append(
            f"only {metrics.n_test} test matches; metrics are very noisy at this size."
        )
    return warnings


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibration_table(y_true: pd.Series, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, per probability bin."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_bins = min(bins, max(2, len(y_true) // 10))

    observed, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    counts, _ = np.histogram(y_prob, bins=edges)

    return pd.DataFrame(
        {
            "predicted": np.round(predicted, 4),
            "observed": np.round(observed, 4),
            "gap": np.round(observed - predicted, 4),
            "n": counts[: len(predicted)],
        }
    )


def plot_calibration(
    results: dict[str, tuple[pd.Series, np.ndarray]],
    out_path: Path,
    title: str = "Calibration",
) -> Path | None:
    """Reliability diagram for one or more models. Returns the path written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in requirements
        log.warning("matplotlib unavailable; skipping calibration plot")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")

    for name, (y_true, y_prob) in results.items():
        table = calibration_table(y_true, y_prob)
        ax.plot(table["predicted"], table["observed"], marker="o", label=name)

    ax.set_xlabel("predicted probability of blue win")
    ax.set_ylabel("observed frequency of blue win")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report(metrics: Metrics, calibration: pd.DataFrame | None = None) -> None:
    print(f"\n=== {metrics.name} ===")
    print(metrics.render())

    if calibration is not None and not calibration.empty:
        print("\n  calibration (quantile bins)")
        print("    predicted  observed     gap     n")
        for _, row in calibration.iterrows():
            print(
                f"    {row['predicted']:>9.3f} {row['observed']:>9.3f} "
                f"{row['gap']:>7.3f} {int(row['n']):>5d}"
            )

    for warning in metrics.warnings:
        print(f"\n  [!] {warning}")


def load_features(path: str | Path) -> pd.DataFrame:
    """Read a feature table, dispatching on file extension.

    CSV is supported because Parquet needs pyarrow's compiled extension, which
    some Windows Application Control policies block.
    """
    path = Path(path)
    table = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    missing = set(feature_columns() + [LABEL_COLUMN]) - set(table.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {sorted(missing)}")
    return table


def run_experiment(
    name: str,
    model: Any,
    table: pd.DataFrame,
    test_fraction: float = 0.25,
    plot_path: Path | None = None,
) -> tuple[Metrics, pd.DataFrame, np.ndarray]:
    """Fit on the earlier matches, score the later ones, report, and plot."""
    split = time_split(table, test_fraction=test_fraction)
    assert_split_is_chronological(split)

    mode = table["leakage_mode"].iloc[0] if "leakage_mode" in table.columns else "unknown"
    log.info(
        "%s | mode=%s | train %d (to %s) | test %d (from %s)",
        name,
        mode,
        split.sizes[0],
        pd.to_datetime(split.cutoff_ts, unit="s"),
        split.sizes[1],
        pd.to_datetime(split.cutoff_ts, unit="s"),
    )

    model.fit(split.X_train, split.y_train)
    y_prob = model.predict_proba(split.X_test)[:, 1]

    metrics = evaluate_predictions(f"{name} [{mode}]", split.y_test, y_prob, split.sizes[0])
    calibration = calibration_table(split.y_test, y_prob)
    report(metrics, calibration)

    if plot_path is not None:
        written = plot_calibration(
            {name: (split.y_test, y_prob)}, plot_path, title=f"{name} ({mode})"
        )
        if written:
            print(f"\n  calibration plot -> {written}")

    return metrics, calibration, y_prob
