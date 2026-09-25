import numpy as np
import pandas as pd
import pytest

from src.model import (
    CANDIDATES,
    ColumnPicker,
    bootstrap_ci,
    build_candidate,
    classification_metrics,
    oof_probabilities,
    risk_band,
    tune_threshold,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"sensor_{i:03d}" for i in range(6)])
    logit = -2.8 + 1.6 * X["sensor_000"] - 1.2 * X["sensor_001"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_classification_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 3
    assert np.isclose(m["recall_fail"], 0.5)
    assert np.isclose(m["specificity"], 0.75)
    assert np.isclose(m["ber"], 1 - (0.5 + 0.75) / 2)
    assert np.isclose(m["precision_fail"], 0.5)


def test_tune_threshold_separates_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    p = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)]
    t = tune_threshold(y, p)
    assert 0.4 < t <= 0.6
    assert classification_metrics(y, p, t)["ber"] == 0


def test_risk_band():
    bands = risk_band(np.array([0.05, 0.3, 0.9]), threshold=0.4)
    assert bands.tolist() == ["Low", "Medium", "High"]


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    lo, hi = bootstrap_ci(y, p, lambda yy, pp: classification_metrics(yy, pp, 0.5)["ber"], n_boot=200)
    point = classification_metrics(y, p, 0.5)["ber"]
    assert lo <= point <= hi


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_every_candidate_learns_signal(data, name):
    X, y = data
    p = oof_probabilities(build_candidate(name), X, y, n_splits=3, n_repeats=1)
    assert p.shape == (len(y),) and np.all((0 <= p) & (p <= 1))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.7, name


def test_column_picker_selects_in_order(data):
    X, _ = data
    out = ColumnPicker(["sensor_003", "sensor_001"]).fit(X).transform(X)
    assert list(out.columns) == ["sensor_003", "sensor_001"]


@pytest.fixture(scope="module")
def timed(data):
    X, y = data
    ts = pd.Series(pd.date_range("2008-07-01", periods=len(y), freq="4h"))
    return X, y, ts


def test_forward_validation_trains_on_past_only(timed):
    from src.model import forward_validation
    X, y, ts = timed
    table, cut = forward_validation(X, y, ts, names=("logistic",), valid_frac=0.3)
    assert list(table["model"]) == ["logistic"]
    assert {"ber", "recall_fail", "roc_auc", "threshold"} <= set(table.columns)
    assert ts.iloc[int(len(ts) * 0.7)] == cut


def test_walk_forward_never_uses_future(timed):
    from src.model import walk_forward
    X, y, ts = timed
    start = ts.iloc[300]
    out = walk_forward("logistic", X, y, ts, start=start, step_days=7)
    assert len(out) == (ts >= start).sum()
    assert (out["train_until"] <= out["timestamp"]).all()
    assert out["fail_prob"].between(0, 1).all()
