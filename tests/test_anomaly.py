import numpy as np
import pandas as pd
import pytest

from src.anomaly import AnomalyDetector


@pytest.fixture(scope="module")
def baseline():
    """Correlated normal data: 8 sensors driven by 2 latent factors + noise."""
    rng = np.random.default_rng(0)
    n = 1500
    f = rng.normal(size=(n, 2))
    load = rng.normal(size=(2, 8))
    X = f @ load + 0.3 * rng.normal(size=(n, 8))
    cols = [f"sensor_{i:03d}" for i in range(8)]
    return pd.DataFrame(X, columns=cols), load, cols


@pytest.fixture(scope="module")
def detector(baseline):
    X, _, _ = baseline
    return AnomalyDetector(variance=0.9, alpha=0.01, random_state=0).fit(X)


def test_pca_keeps_latent_structure(detector):
    assert detector.n_components_ == 2


def test_false_alarm_rate_close_to_alpha(baseline, detector):
    X, load, cols = baseline
    rng = np.random.default_rng(1)
    new = pd.DataFrame(rng.normal(size=(3000, 2)) @ load + 0.3 * rng.normal(size=(3000, 8)), columns=cols)
    s = detector.score(new)
    for flag in ["t2_alarm", "spe_alarm", "iso_alarm"]:
        assert s[flag].mean() < 0.03, flag


def test_broken_correlation_raises_spe_not_t2(baseline, detector):
    X, _, cols = baseline
    x = X.iloc[[0]].copy()
    x["sensor_003"] += 4.0          # off the normal plane
    s = detector.score(x).iloc[0]
    assert s["spe_alarm"]
    assert s["spe"] > detector.spe_limit_ * 3


def test_extreme_along_normal_pattern_raises_t2(baseline, detector):
    X, load, cols = baseline
    x = pd.DataFrame((np.array([[8.0, 0.0]]) @ load), columns=cols)   # far out, but on the plane
    s = detector.score(x).iloc[0]
    assert s["t2_alarm"] and not s["spe_alarm"]


def test_contributions_point_at_the_faulty_sensor(baseline, detector):
    X, _, _ = baseline
    x = X.iloc[[5]].copy()
    x["sensor_006"] += 5.0
    contrib = detector.spe_contributions(x).iloc[0]
    assert contrib.idxmax() == "sensor_006"
    assert detector.score(x).iloc[0]["top_sensors"].startswith("sensor_006")


def test_isolation_forest_flags_gross_outlier(baseline, detector):
    X, _, _ = baseline
    x = X.iloc[[1]].copy() + 25.0
    assert detector.score(x).iloc[0]["iso_alarm"]


def test_limits_come_from_out_of_fold_scores(baseline):
    """Out-of-fold limits must not be smaller than in-sample ones (in-sample points look more normal)."""
    X, _, _ = baseline
    d = AnomalyDetector(variance=0.9, alpha=0.01, random_state=0).fit(X)
    insample_spe_limit = np.quantile(d._spe(X), 0.99)
    assert d.spe_limit_ >= insample_spe_limit * 0.98
