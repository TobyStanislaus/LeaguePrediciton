"""Tests for the evaluation logic: time splitting, metrics, calibration, sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.build_features import LABEL_COLUMN, feature_columns, save_table
from models.evaluate import (
    ALARMING_ACCURACY,
    Metrics,
    SUSPICIOUS_ACCURACY,
    assert_split_is_chronological,
    calibration_table,
    evaluate_predictions,
    leakage_warnings,
    load_features,
    time_split,
)


def make_table(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = feature_columns()
    data = {column: rng.normal(size=n) for column in columns}
    data["match_id"] = [f"EUW1_{i}" for i in range(n)]
    # Deliberately out of order, so the splitter has to sort.
    data["game_start_ts"] = rng.permutation(np.arange(n, dtype=float) * 3600.0)
    data["queue_id"] = 420
    data["game_version"] = "14.18.1"
    data["leakage_mode"] = "reconstructed"
    data[LABEL_COLUMN] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Time splitting
# --------------------------------------------------------------------------


def test_split_is_chronological_not_random():
    table = make_table(100)
    split = time_split(table, test_fraction=0.25)

    ordered = table.sort_values("game_start_ts").reset_index(drop=True)
    train_max = ordered["game_start_ts"].iloc[len(split.X_train) - 1]
    test_min = ordered["game_start_ts"].iloc[len(split.X_train)]
    assert train_max < test_min, "test set must start after the training set ends"


def test_split_sizes_follow_the_fraction():
    split = time_split(make_table(100), test_fraction=0.25)
    assert split.sizes == (75, 25)


def test_split_uses_only_feature_columns():
    split = time_split(make_table(50))
    assert list(split.X_train.columns) == feature_columns()
    assert LABEL_COLUMN not in split.X_train.columns
    assert "game_start_ts" not in split.X_train.columns


def test_split_always_leaves_data_on_both_sides():
    for fraction in (0.01, 0.99):
        split = time_split(make_table(10), test_fraction=fraction)
        assert split.sizes[0] >= 1 and split.sizes[1] >= 1


def test_split_rejects_an_empty_table():
    with pytest.raises(ValueError, match="empty"):
        time_split(make_table(0))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_split_rejects_impossible_fractions(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        time_split(make_table(20), test_fraction=fraction)


def test_chronology_assertion_accepts_a_proper_split():
    assert_split_is_chronological(time_split(make_table(40)))


def test_chronology_assertion_catches_a_shuffled_split():
    """If someone swaps in a random split, this must fail loudly."""
    split = time_split(make_table(40))
    # Simulate a shuffle: the training side now holds the latest matches.
    split.train_ts, split.test_ts = split.test_ts, split.train_ts
    with pytest.raises(AssertionError, match="later than the test set"):
        assert_split_is_chronological(split)


def test_chronology_assertion_rejects_an_empty_side():
    split = time_split(make_table(40))
    split.test_ts = split.test_ts.iloc[0:0]
    with pytest.raises(AssertionError, match="empty"):
        assert_split_is_chronological(split)


def test_split_carries_the_timestamps_of_its_own_rows():
    """The guard is only meaningful if these track the real split."""
    split = time_split(make_table(40), test_fraction=0.25)
    assert len(split.train_ts) == len(split.X_train)
    assert len(split.test_ts) == len(split.X_test)
    assert split.train_ts.max() < split.test_ts.min()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_perfect_predictions_score_perfectly():
    y_true = pd.Series([1, 0, 1, 0] * 25)
    y_prob = np.where(y_true == 1, 0.99, 0.01)
    metrics = evaluate_predictions("perfect", y_true, y_prob, n_train=100)
    assert metrics.accuracy == 1.0
    assert metrics.log_loss < 0.02
    assert metrics.roc_auc == 1.0


def test_coin_flip_predictions_score_like_a_coin_flip():
    y_true = pd.Series([1, 0] * 50)
    y_prob = np.full(100, 0.5)
    metrics = evaluate_predictions("coin", y_true, y_prob, n_train=100)
    assert metrics.log_loss == pytest.approx(np.log(2), abs=1e-6)
    assert metrics.brier == pytest.approx(0.25)


def test_majority_accuracy_reflects_class_imbalance():
    y_true = pd.Series([1] * 80 + [0] * 20)
    metrics = evaluate_predictions("x", y_true, np.full(100, 0.5), n_train=10)
    assert metrics.base_rate == pytest.approx(0.8)
    assert metrics.majority_accuracy == pytest.approx(0.8)


def test_probabilities_are_clipped_so_log_loss_stays_finite():
    y_true = pd.Series([1, 0])
    metrics = evaluate_predictions("x", y_true, np.array([0.0, 1.0]), n_train=10)
    assert np.isfinite(metrics.log_loss)


# --------------------------------------------------------------------------
# The leakage sanity check
# --------------------------------------------------------------------------


def _metrics_with(accuracy: float, n_test: int = 500) -> Metrics:
    return Metrics(
        name="t", accuracy=accuracy, log_loss=0.6, brier=0.24, roc_auc=0.6,
        base_rate=0.5, majority_accuracy=0.5, n_train=1000, n_test=n_test,
    )


def test_plausible_accuracy_raises_no_warning():
    assert leakage_warnings(_metrics_with(0.60)) == []


def test_suspicious_accuracy_is_flagged():
    warnings = leakage_warnings(_metrics_with(SUSPICIOUS_ACCURACY + 0.01))
    assert any("above the expected" in w for w in warnings)


def test_alarming_accuracy_names_leakage_explicitly():
    warnings = leakage_warnings(_metrics_with(ALARMING_ACCURACY + 0.05))
    assert any("leakage bug" in w for w in warnings)


def test_below_chance_accuracy_is_flagged():
    assert any("below chance" in w for w in leakage_warnings(_metrics_with(0.42)))


def test_small_test_sets_are_flagged_as_noisy():
    assert any("noisy" in w for w in leakage_warnings(_metrics_with(0.60, n_test=30)))


def test_ninety_percent_accuracy_would_be_caught():
    """The specific scenario this project must never silently accept."""
    warnings = leakage_warnings(_metrics_with(0.92))
    assert warnings and "leakage bug" in warnings[0]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_table_has_expected_columns():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0.2, 0.8, size=400)
    y_true = pd.Series(rng.binomial(1, y_prob))
    table = calibration_table(y_true, y_prob)
    assert list(table.columns) == ["predicted", "observed", "gap", "n"]
    assert len(table) > 1


def test_a_well_calibrated_model_has_small_gaps():
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0.3, 0.7, size=4000)
    y_true = pd.Series(rng.binomial(1, y_prob))
    table = calibration_table(y_true, y_prob)
    assert table["gap"].abs().max() < 0.12


def test_a_miscalibrated_model_shows_a_large_gap():
    """Predict 0.9 everywhere while the truth is a coin flip."""
    rng = np.random.default_rng(2)
    y_true = pd.Series(rng.binomial(1, 0.5, size=1000))
    y_prob = np.full(1000, 0.9)
    table = calibration_table(y_true, y_prob)
    assert table["gap"].abs().max() > 0.3


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _parquet_available() -> bool:
    """Parquet needs a compiled pyarrow extension, which Windows Application
    Control policies sometimes block. Skip rather than fail in that case."""
    try:
        pd.DataFrame({"a": [1]}).to_parquet
        import pyarrow.parquet  # noqa: F401

        return True
    except Exception:
        return False


FORMATS = ["csv"] + (["parquet"] if _parquet_available() else [])


@pytest.mark.parametrize("suffix", FORMATS)
def test_load_features_rejects_a_table_missing_columns(tmp_path, suffix):
    bad = make_table(10).drop(columns=[feature_columns()[0]])
    path = tmp_path / f"bad.{suffix}"
    save_table(bad, path)
    with pytest.raises(ValueError, match="missing columns"):
        load_features(path)


@pytest.mark.parametrize("suffix", FORMATS)
def test_load_features_round_trips(tmp_path, suffix):
    path = tmp_path / f"good.{suffix}"
    written = save_table(make_table(20), path)
    assert len(load_features(written)) == 20
