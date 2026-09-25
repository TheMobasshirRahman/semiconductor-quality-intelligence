import numpy as np
import pandas as pd
import pytest

from src.feature_selection import (
    SCORERS,
    ConsensusSelector,
    consensus_ranking,
    evaluate_k,
    stability_selection,
)

INFORMATIVE = ["s_a", "s_b", "s_c"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 20)), columns=[f"noise_{i:02d}" for i in range(20)])
    for c in INFORMATIVE:
        X[c] = rng.normal(size=n)
    logit = -3 + 1.5 * X["s_a"] - 1.5 * X["s_b"] + 1.2 * X["s_c"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


@pytest.mark.parametrize("method", ["ttest", "mutual_info", "l1_logistic", "random_forest", "shap"])
def test_each_scorer_ranks_informative_sensors_on_top(data, method):
    X, y = data
    score = SCORERS[method](X, y)
    assert set(score.index) == set(X.columns)
    top5 = score.sort_values(ascending=False).index[:5]
    assert set(INFORMATIVE) <= set(top5), (method, list(top5))


def test_consensus_ranking_columns_and_order(data):
    X, y = data
    r = consensus_ranking(X, y, methods=["ttest", "random_forest"])
    assert {"rank_ttest", "rank_random_forest", "mean_rank", "consensus_rank"} <= set(r.columns)
    assert r["consensus_rank"].is_monotonic_increasing
    assert set(r.index[:3]) == set(INFORMATIVE)


def test_consensus_selector_is_sklearn_transformer(data):
    X, y = data
    sel = ConsensusSelector(k=3, methods=("ttest", "l1_logistic")).fit(X, y)
    out = sel.transform(X)
    assert list(out.columns) == sel.selected_
    assert set(sel.selected_) == set(INFORMATIVE)


def test_stability_selection_frequency(data):
    X, y = data
    freq = stability_selection(X, y, k=3, n_rounds=8, methods=("ttest",), random_state=0)
    assert freq.between(0, 1).all()
    assert (freq[INFORMATIVE] >= 0.75).all()
    assert freq.drop(INFORMATIVE).max() <= 0.5


def test_evaluate_k_returns_metrics_per_k(data):
    X, y = data
    res = evaluate_k(X, y, ks=[3, 10], methods=("ttest",), n_splits=3, random_state=0)
    assert set(res["k"]) == {3, 10}
    assert {"model", "fold", "ber", "recall_fail", "pr_auc", "roc_auc"} <= set(res.columns)
    best = res[res.model == "logistic"].groupby("k")["ber"].mean()
    assert best[3] <= best[10] + 0.05
