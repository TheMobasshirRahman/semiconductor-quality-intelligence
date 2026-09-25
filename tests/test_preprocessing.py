import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    DropConstant,
    DropCorrelated,
    DropHighMissing,
    build_preprocessor,
    time_split,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 100
    a = rng.normal(size=n)
    X = pd.DataFrame({
        "good": a,
        "dup": a * 2 + 0.0001 * rng.normal(size=n),      # ~perfectly correlated with good
        "const": np.ones(n),
        "mostly_nan": np.where(np.arange(n) < 70, np.nan, 1.0) + rng.normal(size=n),
        "some_nan": np.where(np.arange(n) < 5, np.nan, rng.normal(size=n)),
        "other": rng.normal(size=n) * 1000 + 50,
    })
    return X


def test_drop_high_missing(toy):
    out = DropHighMissing(threshold=0.5).fit_transform(toy)
    assert "mostly_nan" not in out.columns
    assert "some_nan" in out.columns


def test_drop_constant(toy):
    out = DropConstant().fit_transform(toy)
    assert "const" not in out.columns
    assert "good" in out.columns


def test_drop_correlated_keeps_first(toy):
    X = toy[["good", "dup", "other"]]
    out = DropCorrelated(threshold=0.95).fit_transform(X)
    assert list(out.columns) == ["good", "other"]


def test_pipeline_output_clean_and_scaled(toy):
    pipe = build_preprocessor()
    out = pipe.fit_transform(toy)
    assert isinstance(out, pd.DataFrame)
    assert not out.isna().any().any()
    assert set(out.columns) == {"good", "some_nan", "other"}
    assert np.allclose(out.mean(), 0, atol=1e-8)
    assert np.allclose(out.std(ddof=0), 1, atol=1e-8)


def test_pipeline_learns_only_from_fit_data(toy):
    """Columns are chosen on train; test gets the same columns even if its stats differ."""
    train, test = toy.iloc[:50].copy(), toy.iloc[50:].copy()
    test["const"] = np.arange(len(test))  # varies in test, constant in train
    pipe = build_preprocessor().fit(train)
    out = pipe.transform(test)
    assert "const" not in out.columns
    assert list(out.columns) == list(pipe.transform(train).columns)


def test_time_split_is_chronological():
    ts = pd.date_range("2008-01-01", periods=10, freq="D")
    df = pd.DataFrame({"timestamp": ts[::-1], "fail": [0, 1] * 5, "x": range(10)})
    train, test = time_split(df, test_size=0.3)
    assert len(train) == 7 and len(test) == 3
    assert train["timestamp"].max() < test["timestamp"].min()


def test_saved_preprocessor_loads_outside_the_script():
    """Running `python -m src.preprocessing` must pickle classes under `src.preprocessing`, not `__main__`."""
    import subprocess
    import sys

    import joblib

    from src.preprocessing import MODELS_DIR

    subprocess.run([sys.executable, "-m", "src.preprocessing"], check=True, capture_output=True)
    pipe = joblib.load(MODELS_DIR / "preprocessor.joblib")
    assert type(pipe.named_steps["drop_missing"]).__module__ == "src.preprocessing"
