"""Data layer for the dashboard: joins the outputs of every pipeline step into one unit table."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Small artifacts the dashboard reads. They are committed to git so the app can be deployed
# without re-running the (slow) pipeline; `python -m src.pipeline` rebuilds all of them.
REQUIRED = {
    "data/raw/secom.data": "(raw UCI SECOM files)",
    "data/processed/cleaning_report.json": "python -m src.preprocessing",
    "data/processed/spc_limits.csv": "python -m src.spc",
    "data/processed/spc_unit_alarms.csv": "python -m src.spc",
    "data/processed/feature_ranking.csv": "python -m src.feature_selection",
    "data/processed/predictions.csv": "python -m src.model",
    "data/processed/anomaly_scores.csv": "python -m src.anomaly",
    "models/fail_model.joblib": "python -m src.model",
    "models/model_card.json": "python -m src.model",
    "models/anomaly_detector.joblib": "python -m src.anomaly",
}


def missing_artifacts():
    """List of (file, command that builds it) for anything not produced yet."""
    return [(f, cmd) for f, cmd in REQUIRED.items() if not (ROOT / f).exists()]


def load_units() -> pd.DataFrame:
    """One row per production unit: raw sensors + label + outputs of Steps 3, 5 and 6."""
    from src.data_loader import load_secom

    units = load_secom()
    units["status"] = np.where(units["fail"] == 1, "Fail", "Pass")

    preds = pd.read_csv(PROCESSED / "predictions.csv")[
        ["unit_id", "period", "fail_prob", "predicted_fail", "risk_band", "source"]]
    preds["period"] = preds["period"].str.title()
    anom = pd.read_csv(PROCESSED / "anomaly_scores.csv")[
        ["unit_id", "t2_ratio", "spe_ratio", "iso_ratio", "mspc_alarm", "iso_alarm", "driver", "top_sensors"]]
    spc = pd.read_csv(PROCESSED / "spc_unit_alarms.csv")[
        ["unit_id", "sensors_in_alarm", "sensors_in_warning", "spc_alarm"]]
    units = units.merge(preds, on="unit_id", how="left").merge(anom, on="unit_id", how="left") \
                 .merge(spc, on="unit_id", how="left")
    units["anomaly_severity"] = units[["t2_ratio", "spe_ratio"]].max(axis=1)
    return units.sort_values("timestamp").reset_index(drop=True)


def load_json(name):
    with open(MODELS / name) as f:
        return json.load(f)


def load_table(name, **kw):
    return pd.read_csv(PROCESSED / name, **kw)


def load_model(name):
    return joblib.load(MODELS / name)


def sensor_columns(df):
    return [c for c in df.columns if c.startswith("sensor_")]


def weekly(units: pd.DataFrame) -> pd.DataFrame:
    """Weekly volume, fails and fail rate."""
    w = units.set_index("timestamp").resample("W-MON", label="left", closed="left")["fail"] \
             .agg(units="count", fails="sum")
    w = w[w["units"] > 0].reset_index().rename(columns={"timestamp": "week"})
    w["fail_rate"] = w["fails"] / w["units"]
    return w


def fail_signature(units: pd.DataFrame, sensors, periods) -> pd.DataFrame:
    """(mean Fail - mean Pass) / std per sensor and period: how the failure fingerprint drifts."""
    out = {}
    for name, mask in periods.items():
        d = units[mask]
        out[name] = [(d.loc[d.fail == 1, s].mean() - d.loc[d.fail == 0, s].mean()) / d[s].std()
                     for s in sensors]
    return pd.DataFrame(out, index=sensors)


def unit_contributions(model, raw_row: pd.DataFrame) -> pd.Series:
    """Logistic model: coefficient x standardised value per selected sensor (log-odds contribution)."""
    clean = model.named_steps["clean"].transform(raw_row)
    x = model.named_steps["select"].transform(clean).iloc[0]
    est = model.named_steps["model"]
    if not hasattr(est, "coef_"):
        return pd.Series(dtype=float)
    return pd.Series(est.coef_[0] * x.to_numpy(), index=x.index)


def score_raw(raw: pd.DataFrame, fail_model, detector, threshold) -> pd.DataFrame:
    """Score new units from raw 590-sensor readings with the saved fail model and anomaly detector."""
    sensors = [f"sensor_{i:03d}" for i in range(590)]
    missing = [s for s in sensors if s not in raw.columns]
    if missing:
        raise ValueError(f"{len(missing)} sensor columns missing, e.g. {missing[:3]}")
    X = raw[sensors].apply(pd.to_numeric, errors="coerce")
    prob = fail_model.predict_proba(X)[:, 1]
    clean = fail_model.named_steps["clean"].transform(X)
    anom = detector.score(clean)
    from src.model import risk_band
    out = pd.DataFrame({
        "fail_prob": prob.round(4),
        "risk_band": risk_band(prob, threshold),
        "anomaly_alarm": anom["mspc_alarm"].to_numpy(),
        "anomaly_severity": anom[["t2_ratio", "spe_ratio"]].max(axis=1).round(2).to_numpy(),
        "top_sensors": anom["top_sensors"].to_numpy(),
    }, index=raw.index)
    if "unit_id" in raw.columns:
        out.insert(0, "unit_id", raw["unit_id"])
    return out
