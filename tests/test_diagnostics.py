"""Tests for the three-way split and the monitoring curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build_features import LABEL_COLUMN, feature_columns
from models.diagnostics import learning_curve, plot_diagnostics
from models.evaluate import (
    assert_3way_is_chronological,
    calibration_table,
    time_split_3way,
)


def make_table(n: int = 300, seed: int = 0, signal: float = 0.0) -> pd.DataFrame:
    """Synthetic feature table; ``signal`` controls how learnable the label is."""
    rng = np.random.default_rng(seed)
    columns = feature_columns()
    data = {c: rng.normal(size=n) for c in columns}
    frame = pd.DataFrame(data)
    logit = signal * frame["diff_rank_points_mean"]
    frame[LABEL_COLUMN] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame["match_id"] = [f"m{i}" for i in range(n)]
    frame["game_start_ts"] = rng.permutation(np.arange(n, dtype=float) * 3600.0)
    frame["queue_id"] = 420
    frame["game_version"] = "14.18.1"
    frame["leakage_mode"] = "reconstructed"
    return frame


def simple_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([("impute", SimpleImputer()), ("clf", LogisticRegression(max_iter=1000))])


# --------------------------------------------------------------------------
# Three-way split
# --------------------------------------------------------------------------


def test_default_split_is_sixty_twenty_twenty():
    split = time_split_3way(make_table(300))
    assert split.sizes == (180, 60, 60)


def test_split_is_ordered_train_then_validation_then_test():
    split = time_split_3way(make_table(300))
    assert split.train_ts.max() < split.val_ts.min()
    assert split.val_ts.max() < split.test_ts.min()
    assert_3way_is_chronological(split)


def test_chronology_assertion_catches_training_reaching_forward():
    split = time_split_3way(make_table(300))
    split.train_ts, split.val_ts = split.val_ts, split.train_ts
    with pytest.raises(AssertionError, match="validation period"):
        assert_3way_is_chronological(split)


def test_chronology_assertion_catches_validation_reaching_into_test():
    split = time_split_3way(make_table(300))
    split.val_ts, split.test_ts = split.test_ts, split.val_ts
    with pytest.raises(AssertionError, match="test period"):
        assert_3way_is_chronological(split)


def test_split_uses_only_feature_columns():
    split = time_split_3way(make_table(120))
    for frame in (split.X_train, split.X_val, split.X_test):
        assert list(frame.columns) == feature_columns()


def test_custom_proportions_are_respected():
    split = time_split_3way(make_table(200), val_fraction=0.1, test_fraction=0.3)
    assert split.sizes == (120, 20, 60)


@pytest.mark.parametrize("val,test", [(0.6, 0.6), (0.0, 0.0), (-0.1, 0.2)])
def test_impossible_proportions_are_rejected(val, test):
    with pytest.raises(ValueError):
        time_split_3way(make_table(100), val_fraction=val, test_fraction=test)


def test_tiny_tables_are_rejected():
    with pytest.raises(ValueError, match="at least three"):
        time_split_3way(make_table(2))


# --------------------------------------------------------------------------
# Learning curve
# --------------------------------------------------------------------------


def test_learning_curve_grows_the_training_set_monotonically():
    split = time_split_3way(make_table(300))
    curve = learning_curve(simple_model, split, steps=5, min_train=40)
    assert curve["train_size"].is_monotonic_increasing
    assert curve["train_size"].iloc[-1] == len(split.X_train)


def test_learning_curve_reports_both_train_and_validation():
    split = time_split_3way(make_table(300))
    curve = learning_curve(simple_model, split, steps=4, min_train=40)
    assert {"train_size", "val_accuracy", "val_log_loss", "train_accuracy"} <= set(curve.columns)
    assert (curve["val_accuracy"].between(0, 1)).all()


def test_learning_curve_uses_chronological_prefixes_not_random_samples():
    """Each step must train on the oldest N matches, never a shuffle."""
    seen: list[pd.Index] = []

    class Recorder:
        def fit(self, X, y):
            seen.append(X.index)
            self._p = float(y.mean())
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    split = time_split_3way(make_table(300))
    learning_curve(Recorder, split, steps=4, min_train=40)

    for index in seen:
        assert list(index) == list(range(len(index))), "prefix was not the oldest rows"
    assert all(len(seen[i]) < len(seen[i + 1]) for i in range(len(seen) - 1))


def test_learning_curve_needs_a_minimum_training_set():
    split = time_split_3way(make_table(60))
    with pytest.raises(ValueError, match="at least"):
        learning_curve(simple_model, split, min_train=500)


def test_a_learnable_signal_produces_a_rising_curve():
    """Sanity check that the curve can detect improvement at all."""
    split = time_split_3way(make_table(600, signal=2.0))
    curve = learning_curve(simple_model, split, steps=6, min_train=40)
    assert curve["val_accuracy"].iloc[-1] > curve["val_accuracy"].iloc[0]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def test_plot_writes_a_file(tmp_path):
    split = time_split_3way(make_table(300))
    curve = learning_curve(simple_model, split, steps=4, min_train=40)
    calibration = calibration_table(split.y_test, np.full(len(split.y_test), 0.5))

    out = plot_diagnostics(curve, None, calibration, tmp_path / "d.png", subtitle="test")
    assert out.exists()
    assert out.stat().st_size > 5000


def test_plot_survives_missing_optional_panels(tmp_path):
    split = time_split_3way(make_table(300))
    curve = learning_curve(simple_model, split, steps=3, min_train=40)
    out = plot_diagnostics(curve, None, None, tmp_path / "d2.png")
    assert out.exists()
