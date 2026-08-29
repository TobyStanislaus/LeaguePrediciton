"""Tests for the paired bootstrap used to put error bars on the leakage gap."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_report():
    path = REPO_ROOT / "scripts" / "leakage_report.py"
    spec = importlib.util.spec_from_file_location("leakage_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return _load_report()


def _labels(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n)


def test_identical_predictions_give_a_zero_gap(report):
    y = _labels()
    prob = np.full(len(y), 0.5)
    stats = report.paired_bootstrap(y, prob, prob, iterations=300)
    assert stats["accuracy_gap"] == pytest.approx(0.0)
    assert stats["accuracy_lo"] <= 0 <= stats["accuracy_hi"]


def test_a_perfect_leak_is_detected_with_an_interval_clear_of_zero(report):
    """One model sees the answer, the other guesses -- the gap must be obvious."""
    y = _labels()
    leaky = np.where(y == 1, 0.99, 0.01)
    rng = np.random.default_rng(1)
    clean = rng.uniform(0.4, 0.6, size=len(y))

    stats = report.paired_bootstrap(y, leaky, clean, iterations=300)
    assert stats["accuracy_gap"] > 0.3
    assert stats["accuracy_lo"] > 0, "interval should exclude zero for a blatant leak"
    assert stats["share_positive"] == 1.0


def test_the_interval_widens_as_the_test_set_shrinks(report):
    """The whole point: a small test set cannot resolve a small gap."""
    def width(n: int) -> float:
        y = _labels(n, seed=3)
        rng = np.random.default_rng(4)
        clean = rng.uniform(0.3, 0.7, size=n)
        leaky = np.clip(clean + 0.05 * (2 * y - 1), 0.01, 0.99)
        stats = report.paired_bootstrap(y, leaky, clean, iterations=400, random_state=2)
        return stats["accuracy_hi"] - stats["accuracy_lo"]

    assert width(80) > width(1600)


def test_gap_is_reported_in_the_direction_of_the_leaky_model(report):
    y = _labels()
    leaky = np.where(y == 1, 0.9, 0.1)
    clean = np.where(y == 1, 0.1, 0.9)  # deliberately inverted
    stats = report.paired_bootstrap(y, leaky, clean, iterations=200)
    assert stats["accuracy_gap"] > 0
    assert stats["auc_gap"] > 0


def test_bootstrap_is_reproducible(report):
    y = _labels()
    rng = np.random.default_rng(5)
    clean = rng.uniform(0.3, 0.7, size=len(y))
    leaky = np.clip(clean + 0.1 * (2 * y - 1), 0.01, 0.99)

    first = report.paired_bootstrap(y, leaky, clean, iterations=200, random_state=11)
    second = report.paired_bootstrap(y, leaky, clean, iterations=200, random_state=11)
    assert first == second
