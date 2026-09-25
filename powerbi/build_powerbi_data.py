"""
Builds Power BI ready CSV files from the raw SECOM dataset.

Output (in this folder):
  secom_units.csv            - one row per production unit (main fact table)
  secom_sensor_readings.csv  - long format readings of top sensors + SPC limits
  secom_sensor_summary.csv   - one row per sensor (quality + importance stats)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
RAW = HERE.parent / "data" / "raw"
TOP_N = 20          # sensors exported for SPC / long table
TOP_IN_UNITS = 10   # sensors also added as columns in the units table
SEED = 42

# ---------- 1. Load + merge ----------
X = pd.read_csv(RAW / "secom.data", sep=" ", header=None)
X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

labels = pd.read_csv(RAW / "secom_labels.data", sep=" ", header=None,
                     names=["label", "timestamp"])
labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
y = (labels["label"] == 1).astype(int)  # 1 = Fail, 0 = Pass

unit_id = [f"U{i + 1:04d}" for i in range(len(X))]

# ---------- 2. Data quality ----------
missing_pct = X.isna().mean() * 100
is_constant = X.nunique(dropna=True) <= 1
high_missing = missing_pct > 50
keep = ~(is_constant | high_missing)
Xk = X.loc[:, keep]
Xi = Xk.fillna(Xk.median())

# ---------- 3. Feature importance + out-of-fold fail probability ----------
rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                            min_samples_leaf=3, random_state=SEED, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fail_prob = cross_val_predict(rf, Xi, y, cv=cv, method="predict_proba")[:, 1]
rf.fit(Xi, y)
importance = pd.Series(rf.feature_importances_, index=Xi.columns).sort_values(ascending=False)
top_sensors = importance.index[:TOP_N].tolist()

# ---------- 4. Anomaly detection ----------
Xs = StandardScaler().fit_transform(Xi)
iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=SEED).fit(Xs)
anomaly_score = -iso.score_samples(Xs)  # higher = more abnormal
is_anomaly = iso.predict(Xs) == -1

# ---------- 5. SPC limits (baseline = Pass units) ----------
base = Xk.loc[y == 0, top_sensors]
center = base.mean()
sigma = base.std()
ucl, lcl = center + 3 * sigma, center - 3 * sigma
ooc = (Xk[top_sensors] > ucl) | (Xk[top_sensors] < lcl)  # NaN compares False

# ---------- 6. Units table ----------
p_hi, p_med = np.quantile(fail_prob, [0.90, 0.70])
risk = np.select([fail_prob >= p_hi, fail_prob >= p_med], ["High", "Medium"], "Low")

units = pd.DataFrame({
    "unit_id": unit_id,
    "timestamp": labels["timestamp"],
    "date": labels["timestamp"].dt.date,
    "week_start": (labels["timestamp"] - pd.to_timedelta(labels["timestamp"].dt.weekday, unit="D")).dt.date,
    "month": labels["timestamp"].dt.strftime("%Y-%m"),
    "day_name": labels["timestamp"].dt.day_name(),
    "hour": labels["timestamp"].dt.hour,
    "shift": pd.cut(labels["timestamp"].dt.hour, [-1, 7, 15, 23],
                    labels=["Night (00-08)", "Morning (08-16)", "Evening (16-24)"]),
    "status": np.where(y == 1, "Fail", "Pass"),
    "fail_flag": y,
    "missing_sensor_count": X.isna().sum(axis=1),
    "predicted_fail_prob": fail_prob.round(4),
    "risk_level": risk,
    "anomaly_score": anomaly_score.round(4),
    "is_anomaly": np.where(is_anomaly, "Yes", "No"),
    "ooc_sensor_count": ooc.sum(axis=1),
})
for s in top_sensors[:TOP_IN_UNITS]:
    units[s] = X[s]
units.to_csv(HERE / "secom_units.csv", index=False)

# ---------- 7. Long sensor readings (top sensors) ----------
long = (X[top_sensors].assign(unit_id=unit_id)
        .melt(id_vars="unit_id", var_name="sensor", value_name="value"))
long = long.merge(units[["unit_id", "timestamp", "date", "status"]], on="unit_id")
long["center_line"] = long["sensor"].map(center.round(4))
long["ucl"] = long["sensor"].map(ucl.round(4))
long["lcl"] = long["sensor"].map(lcl.round(4))
long["out_of_control"] = np.where(
    long["value"].notna() & ((long["value"] > long["ucl"]) | (long["value"] < long["lcl"])),
    "Yes", "No")
long["importance_rank"] = long["sensor"].map({s: i + 1 for i, s in enumerate(top_sensors)})
long = long.sort_values(["importance_rank", "timestamp"])
long.to_csv(HERE / "secom_sensor_readings.csv", index=False)

# ---------- 8. Sensor summary ----------
rank = pd.Series(range(1, len(importance) + 1), index=importance.index)
summary = pd.DataFrame({
    "sensor": X.columns,
    "missing_pct": missing_pct.round(2).values,
    "is_constant": np.where(is_constant, "Yes", "No"),
    "status": np.select([is_constant, high_missing], ["Dropped - constant", "Dropped - >50% missing"], "Kept"),
    "mean": X.mean().round(4).values,
    "std": X.std().round(4).values,
    "pass_mean": X[y == 0].mean().round(4).values,
    "fail_mean": X[y == 1].mean().round(4).values,
})
summary["importance"] = summary["sensor"].map(importance).round(5)
summary["importance_rank"] = summary["sensor"].map(rank).astype("Int64")
summary["is_top_sensor"] = np.where(summary["sensor"].isin(top_sensors), "Yes", "No")
summary.to_csv(HERE / "secom_sensor_summary.csv", index=False)

print("units:", units.shape, "| readings:", long.shape, "| summary:", summary.shape)
print("kept sensors:", int(keep.sum()), "| top sensors:", top_sensors[:10])
print("risk thresholds  high>=%.3f  medium>=%.3f" % (p_hi, p_med))
print("fail rate by risk:\n", units.groupby("risk_level")["fail_flag"].agg(["count", "mean"]))
