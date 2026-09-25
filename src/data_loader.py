"""Load the raw SECOM files and merge sensors with labels into one DataFrame."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_sensors(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """590 sensor columns, named sensor_000 ... sensor_589."""
    X = pd.read_csv(raw_dir / "secom.data", sep=" ", header=None)
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]
    return X


def load_labels(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Timestamp + target. In the raw file -1 = Pass, 1 = Fail."""
    y = pd.read_csv(raw_dir / "secom_labels.data", sep=" ", header=None,
                    names=["label", "timestamp"])
    y["timestamp"] = pd.to_datetime(y["timestamp"], format="%d/%m/%Y %H:%M:%S")
    y["fail"] = (y["label"] == 1).astype(int)
    return y[["timestamp", "fail"]]


def load_secom(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per production unit: unit_id, timestamp, fail, sensor_000..sensor_589."""
    X = load_sensors(raw_dir)
    y = load_labels(raw_dir)
    df = pd.concat([y, X], axis=1)
    df.insert(0, "unit_id", [f"U{i + 1:04d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_secom()
    out = PROCESSED_DIR / "secom_merged.csv"
    df.to_csv(out, index=False)
    print(f"Saved {df.shape} -> {out}")
