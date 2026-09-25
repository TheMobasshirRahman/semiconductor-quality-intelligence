"""
Cleaning pipeline for SECOM sensor data.

Every step is a scikit-learn transformer, so all decisions (which sensors to drop,
median values, scaling) are learned on the TRAIN period only and then replayed on
the test period -> no data leakage.

    DropHighMissing -> DropConstant -> median impute -> DropCorrelated -> StandardScaler

Run as a script to build the processed train/test files:
    python -m src.preprocessing
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROCESSED_DIR, PROJECT_ROOT, load_secom

MODELS_DIR = PROJECT_ROOT / "models"


class _ColumnSelector(BaseEstimator, TransformerMixin):
    """Base class: subclasses set `self.keep_` (list of column names) in fit()."""

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


class DropHighMissing(_ColumnSelector):
    """Drop sensors whose share of missing values is above `threshold`."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.missing_share_ = X.isna().mean()
        self.keep_ = self.missing_share_.index[self.missing_share_ <= self.threshold].tolist()
        self.dropped_ = self.missing_share_.index[self.missing_share_ > self.threshold].tolist()
        return self


class DropConstant(_ColumnSelector):
    """Drop sensors with a single distinct value (zero information)."""

    def fit(self, X, y=None):
        nunique = X.nunique(dropna=True)
        self.keep_ = nunique.index[nunique > 1].tolist()
        self.dropped_ = nunique.index[nunique <= 1].tolist()
        return self


class DropCorrelated(_ColumnSelector):
    """Drop near-duplicate sensors: for each pair with |r| > threshold keep the first one."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = upper.columns[(upper > self.threshold).any()]
        self.dropped_ = to_drop.tolist()
        self.keep_ = [c for c in X.columns if c not in set(to_drop)]
        return self


def build_preprocessor(missing_threshold=0.5, corr_threshold=0.95) -> Pipeline:
    pipe = Pipeline([
        ("drop_missing", DropHighMissing(threshold=missing_threshold)),
        ("drop_constant", DropConstant()),
        ("impute", SimpleImputer(strategy="median")),
        ("drop_correlated", DropCorrelated(threshold=corr_threshold)),
        ("scale", StandardScaler()),
    ])
    return pipe.set_output(transform="pandas")


def time_split(df: pd.DataFrame, test_size=0.25, time_col="timestamp"):
    """Chronological split: the latest `test_size` share of units becomes the test set."""
    df = df.sort_values(time_col).reset_index(drop=True)
    cut = int(round(len(df) * (1 - test_size)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def cleaning_report(pipe: Pipeline, n_raw: int, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    steps = pipe.named_steps
    return {
        "raw_sensors": n_raw,
        "dropped_high_missing": len(steps["drop_missing"].dropped_),
        "dropped_constant": len(steps["drop_constant"].dropped_),
        "dropped_correlated": len(steps["drop_correlated"].dropped_),
        "final_sensors": len(steps["drop_correlated"].keep_),
        "train": {
            "units": len(train),
            "fails": int(train["fail"].sum()),
            "fail_rate": round(float(train["fail"].mean()), 4),
            "from": str(train["timestamp"].min()),
            "to": str(train["timestamp"].max()),
        },
        "test": {
            "units": len(test),
            "fails": int(test["fail"].sum()),
            "fail_rate": round(float(test["fail"].mean()), 4),
            "from": str(test["timestamp"].min()),
            "to": str(test["timestamp"].max()),
        },
        "dropped_sensors": {
            "high_missing": steps["drop_missing"].dropped_,
            "constant": steps["drop_constant"].dropped_,
            "correlated": steps["drop_correlated"].dropped_,
        },
    }


def main(test_size=0.25):
    df = load_secom()
    meta_cols = ["unit_id", "timestamp", "fail"]
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    train, test = time_split(df, test_size=test_size)

    pipe = build_preprocessor()
    X_train = pipe.fit_transform(train[sensor_cols])
    X_test = pipe.transform(test[sensor_cols])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([train[meta_cols], X_train], axis=1).to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
    pd.concat([test[meta_cols], X_test.set_index(test.index)], axis=1).to_csv(
        PROCESSED_DIR / "test_clean.csv", index=False)
    joblib.dump(pipe, MODELS_DIR / "preprocessor.joblib")

    report = cleaning_report(pipe, len(sensor_cols), train, test)
    with open(PROCESSED_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Sensors: {report['raw_sensors']} raw "
          f"-> -{report['dropped_high_missing']} high-missing "
          f"-> -{report['dropped_constant']} constant "
          f"-> -{report['dropped_correlated']} correlated "
          f"= {report['final_sensors']} final")
    for part in ("train", "test"):
        r = report[part]
        print(f"{part:5}: {r['units']} units, {r['fails']} fails ({r['fail_rate']:.1%}), "
              f"{r['from'][:10]} -> {r['to'][:10]}")
    return report


if __name__ == "__main__":
    # import via the package so pickled transformers reference `src.preprocessing`, not `__main__`
    from src.preprocessing import main as _main
    _main()
