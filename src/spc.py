"""
Statistical Process Control (SPC) for SECOM sensors.

- Individuals control charts: one measurement per production unit.
- Phase I: limits are learned on known-good units (Pass) of the TRAIN period.
  Phase II: every unit is monitored in time order with those fixed limits.
- Two ways to build limits:
    * ControlLimits / fit_imr_limits -> textbook I-MR chart, mean ± 3σ with
      sigma = median moving range / 0.954. Assumes independent, normal data.
    * PercentileLimits -> empirical-percentile limits for non-normal data
      (ISO 22514 style). Each reading is turned into an equivalent z-score via
      the baseline's empirical CDF, so "3σ" means the same tail probability
      (0.135%) whatever the shape of the distribution.
  SECOM sensors are heavily skewed and autocorrelated, so PercentileLimits is
  the default (see notebooks/03_spc_monitoring.ipynb for the comparison).
- Western Electric rules on the z-scores:
    rule1 + rule2 -> out-of-control ALARM for the unit
    rule3 + rule4 -> shift / trend WARNING (run rules over-fire on autocorrelated data)

Run as a script:
    python -m src.spc
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_loader import PROCESSED_DIR, load_secom
from src.preprocessing import MODELS_DIR, time_split

MEDIAN_MR_TO_SIGMA = 0.954   # d4 constant for median moving range, n=2
RULES = ["rule1", "rule2", "rule3", "rule4"]
RULE_NAMES = {
    "rule1": "1 point beyond 3σ",
    "rule2": "2 of 3 beyond 2σ (same side)",
    "rule3": "4 of 5 beyond 1σ (same side)",
    "rule4": "8 in a row on one side",
}
ALARM_RULES = ["rule1", "rule2"]
WARNING_RULES = ["rule3", "rule4"]


@dataclass
class ControlLimits:
    """Classic Shewhart limits: center ± k·sigma."""
    sensor: str
    center: float
    sigma: float

    def level(self, k):
        return self.center + k * self.sigma

    @property
    def ucl(self):
        return self.level(3)

    @property
    def lcl(self):
        return self.level(-3)

    def to_z(self, x: pd.Series) -> pd.Series:
        return (x - self.center) / self.sigma if self.sigma > 0 else x * 0.0


@dataclass
class PercentileLimits:
    """Empirical-percentile limits for non-normal data; `level(k)` is the value at Φ(k)."""
    sensor: str
    baseline: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, x: pd.Series, sensor: str) -> "PercentileLimits":
        return cls(sensor=sensor, baseline=np.sort(x.dropna().to_numpy(dtype=float)))

    def level(self, k):
        return float(np.quantile(self.baseline, norm.cdf(k)))

    @property
    def center(self):
        return self.level(0)

    @property
    def ucl(self):
        return self.level(3)

    @property
    def lcl(self):
        return self.level(-3)

    def to_z(self, x: pd.Series) -> pd.Series:
        b, n = self.baseline, len(self.baseline)
        v = x.to_numpy(dtype=float)
        lo, hi = np.searchsorted(b, v, "left"), np.searchsorted(b, v, "right")
        rank = np.clip((lo + hi) / 2, 0.5, n - 0.5)       # mid-rank, bounded inside (0, 1)
        z = pd.Series(norm.ppf(rank / n), index=x.index)
        return z.where(x.notna())


def fit_imr_limits(x: pd.Series, sensor: str, max_iter: int = 3) -> ControlLimits:
    """Phase I for an I-MR chart: estimate center/sigma, drop points beyond 3σ, re-estimate."""
    x = x.dropna()
    for _ in range(max_iter):
        mr = x.diff().abs().dropna()
        sigma = mr.median() / MEDIAN_MR_TO_SIGMA
        center = x.mean()
        if sigma == 0:
            break
        inside = (x - center).abs() <= 3 * sigma
        if inside.all():
            break
        x = x[inside]
    return ControlLimits(sensor=sensor, center=float(center), sigma=float(sigma))


def western_electric(x: pd.Series, lim) -> pd.DataFrame:
    """Boolean flags per point for the 4 Western Electric rules. Missing readings are skipped."""
    z = lim.to_z(x).dropna()

    def k_of_n(cond, k, n):
        return cond.astype(int).rolling(n, min_periods=n).sum() >= k

    out = pd.DataFrame(index=z.index)
    out["rule1"] = z.abs() > 3
    out["rule2"] = k_of_n(z > 2, 2, 3) | k_of_n(z < -2, 2, 3)
    out["rule3"] = k_of_n(z > 1, 4, 5) | k_of_n(z < -1, 4, 5)
    out["rule4"] = k_of_n(z > 0, 8, 8) | k_of_n(z < 0, 8, 8)
    out = out.reindex(x.index, fill_value=False).astype(bool)
    out["alarm"] = out[ALARM_RULES].any(axis=1)
    out["warning"] = out[WARNING_RULES].any(axis=1)
    out["any_violation"] = out["alarm"] | out["warning"]
    return out


def select_monitoring_sensors(train: pd.DataFrame, candidates, n=10) -> pd.Series:
    """Rank sensors by standardised Pass/Fail mean difference on the train period."""
    X, y = train[candidates], train["fail"]
    smd = ((X[y == 1].mean() - X[y == 0].mean()) / X.std()).abs()
    return smd.sort_values(ascending=False).head(n)


def load_monitoring_frame(test_size=0.25):
    """Raw (unscaled) data in time order + train/test boundary + cleaned sensor list."""
    df = load_secom()
    train, test = time_split(df, test_size=test_size)
    df = pd.concat([train, test], ignore_index=True)
    df["period"] = np.where(df.index >= len(train), "test", "train")
    pipe = joblib.load(MODELS_DIR / "preprocessor.joblib")
    candidates = pipe.named_steps["drop_correlated"].keep_
    return df, candidates


def run_spc(n_sensors=10, test_size=0.25, method="percentile"):
    df, candidates = load_monitoring_frame(test_size)
    train = df[df["period"] == "train"]
    baseline = train[train["fail"] == 0]
    ranked = select_monitoring_sensors(train, candidates, n=n_sensors)

    fit = PercentileLimits.fit if method == "percentile" else fit_imr_limits
    limits, flags = [], {}
    for rank, sensor in enumerate(ranked.index, start=1):
        lim = fit(baseline[sensor], sensor)
        f = western_electric(df[sensor], lim)
        flags[sensor] = f
        test_pass = (df["period"] == "test") & (df["fail"] == 0)
        limits.append({
            "rank": rank,
            "sensor": sensor,
            "method": method,
            "lcl": lim.lcl, "center": lim.center, "ucl": lim.ucl,
            "zone_minus2": lim.level(-2), "zone_minus1": lim.level(-1),
            "zone_plus1": lim.level(1), "zone_plus2": lim.level(2),
            "pass_fail_smd": round(float(ranked[sensor]), 4),
            **{f"{r}_count": int(f[r].sum()) for r in RULES},
            "alarm_rate_baseline": round(float(f.loc[baseline.index, "alarm"].mean()), 4),
            "alarm_rate_test_pass": round(float(f.loc[test_pass, "alarm"].mean()), 4),
            "warning_rate_test_pass": round(float(f.loc[test_pass, "warning"].mean()), 4),
        })
    limits = pd.DataFrame(limits)

    units = df[["unit_id", "timestamp", "period", "fail"]].copy()
    units["sensors_in_alarm"] = sum(f["alarm"].astype(int) for f in flags.values())
    units["sensors_in_warning"] = sum(f["warning"].astype(int) for f in flags.values())
    units["spc_alarm"] = units["sensors_in_alarm"] > 0

    events = []
    for sensor, f in flags.items():
        for idx, row in f[f["any_violation"]].iterrows():
            events.append({
                "unit_id": df.at[idx, "unit_id"], "timestamp": df.at[idx, "timestamp"],
                "period": df.at[idx, "period"], "sensor": sensor, "value": df.at[idx, sensor],
                "fail": int(df.at[idx, "fail"]),
                "severity": "alarm" if row["alarm"] else "warning",
                "rules": ", ".join(r for r in RULES if row[r]),
            })
    events = pd.DataFrame(events).sort_values(["timestamp", "sensor"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    limits.to_csv(PROCESSED_DIR / "spc_limits.csv", index=False)
    units.to_csv(PROCESSED_DIR / "spc_unit_alarms.csv", index=False)
    events.to_csv(PROCESSED_DIR / "spc_violations.csv", index=False)
    return limits, units, events


def alarm_effectiveness(units: pd.DataFrame) -> pd.DataFrame:
    """Fail rate of units with vs without an SPC alarm, per period."""
    return (units.groupby(["period", "spc_alarm"])["fail"]
            .agg(units="count", fails="sum", fail_rate="mean")
            .reset_index())


if __name__ == "__main__":
    limits, units, events = run_spc()
    cols = ["rank", "sensor", "lcl", "center", "ucl",
            "alarm_rate_baseline", "alarm_rate_test_pass", "warning_rate_test_pass"]
    print(limits[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(alarm_effectiveness(units).round(3).to_string(index=False))
