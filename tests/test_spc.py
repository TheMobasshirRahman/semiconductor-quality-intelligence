import numpy as np
import pandas as pd

from src.spc import ControlLimits, PercentileLimits, fit_imr_limits, western_electric


def limits(cl=0.0, sigma=1.0):
    return ControlLimits(sensor="s", center=cl, sigma=sigma)


def test_control_limits_are_center_plus_minus_3_sigma():
    lim = limits(10, 2)
    assert lim.ucl == 16 and lim.lcl == 4


def test_imr_sigma_from_moving_range():
    # alternating 0, 1 -> every moving range = 1 -> sigma = median(MR)/0.954
    x = pd.Series([0.0, 1.0] * 20)
    lim = fit_imr_limits(x, "s")
    assert np.isclose(lim.center, 0.5)
    assert np.isclose(lim.sigma, 1 / 0.954)


def test_imr_limits_ignore_nan():
    x = pd.Series([0.0, np.nan, 1.0, 0.0, 1.0, np.nan, 0.0])
    lim = fit_imr_limits(x, "s")
    assert np.isfinite(lim.center) and np.isfinite(lim.sigma)


def test_rule1_point_beyond_3_sigma():
    x = pd.Series([0, 0.5, -0.5, 3.5, 0, -3.2])
    r = western_electric(x, limits())
    assert r["rule1"].tolist() == [False, False, False, True, False, True]


def test_rule2_two_of_three_beyond_2_sigma_same_side():
    x = pd.Series([0, 2.5, 0.1, 2.2, 0, -2.5, 2.5])
    r = western_electric(x, limits())
    # index 3: window (2.5, 0.1, 2.2) -> 2 above +2σ
    assert r["rule2"].tolist() == [False, False, False, True, False, False, False]


def test_rule3_four_of_five_beyond_1_sigma_same_side():
    x = pd.Series([1.5, 1.2, 0.0, 1.8, 1.1, -1.5])
    r = western_electric(x, limits())
    assert r["rule3"].tolist() == [False, False, False, False, True, False]


def test_rule4_eight_in_a_row_same_side():
    x = pd.Series([0.1] * 8 + [-0.1])
    r = western_electric(x, limits())
    assert r["rule4"].tolist() == [False] * 7 + [True, False]


def test_nan_points_are_skipped_not_flagged():
    x = pd.Series([0.1, 0.1, np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0])
    r = western_electric(x, limits())
    assert not r.loc[2].any()
    assert r.loc[8, "rule4"]           # 8 non-missing points above center
    assert r.loc[9, "rule1"]
    assert r.loc[9, "any_violation"]


def test_percentile_limits_median_maps_to_zero_and_extremes_alarm():
    base = pd.Series(np.arange(1, 1001, dtype=float))
    lim = PercentileLimits.fit(base, "s")
    z = lim.to_z(pd.Series([500.5, 5000.0, -10.0, np.nan]))
    assert abs(z[0]) < 0.01
    assert z[1] > 3 and z[2] < -3
    assert np.isnan(z[3])
    assert lim.lcl < lim.center < lim.ucl


def test_percentile_limits_false_alarm_rate_on_skewed_data():
    """Skewed (lognormal) in-control data: ~0.27% beyond limits, unlike mean ± 3σ."""
    rng = np.random.default_rng(1)
    base = pd.Series(rng.lognormal(0, 1, 20000))
    new = pd.Series(rng.lognormal(0, 1, 20000))
    lim = PercentileLimits.fit(base, "s")
    rate = western_electric(new, lim)["rule1"].mean()
    assert rate < 0.006
    naive = ((new - base.mean()).abs() > 3 * base.std()).mean()
    assert naive > rate
