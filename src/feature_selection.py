"""
Feature selection: which of the 272 cleaned sensors drive yield failures?

Five scoring methods, each looking at the data differently:
    ttest          |Welch t| Pass vs Fail            -> univariate, linear shift in mean
    mutual_info    mutual information                -> univariate, any (non-linear) dependence
    l1_logistic    |coef| of L1 logistic regression  -> multivariate, sparse, linear
    random_forest  impurity importance               -> multivariate, non-linear, interactions
    shap           mean |SHAP| of a random forest    -> multivariate, model-explanation view

They are combined into a CONSENSUS ranking (mean rank). Two extra checks:
    stability_selection -> how often a sensor stays in the top-k over bootstrap subsamples
    evaluate_k          -> nested CV: ranking is re-done inside every fold, so the chosen K
                           is not flattered by selection bias

Everything uses the TRAIN period only. Run as a script:
    python -m src.feature_selection
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROCESSED_DIR
from src.preprocessing import MODELS_DIR

SEED = 42
FAST_METHODS = ("ttest", "mutual_info", "l1_logistic", "random_forest")
ALL_METHODS = FAST_METHODS + ("shap",)


# ---------------------------------------------------------------- scorers
def _rf(random_state=SEED, n_estimators=300):
    return RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",
                                  min_samples_leaf=3, n_jobs=-1, random_state=random_state)


def score_ttest(X, y, random_state=SEED):
    t, _ = ttest_ind(X[y == 1], X[y == 0], equal_var=False, nan_policy="omit")
    return pd.Series(np.nan_to_num(np.abs(t)), index=X.columns)


def score_mutual_info(X, y, random_state=SEED):
    return pd.Series(mutual_info_classif(X, y, n_neighbors=3, random_state=random_state), index=X.columns)


def score_l1_logistic(X, y, random_state=SEED, C=0.1):
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced",
                            random_state=random_state).fit(Xs, y)
    return pd.Series(np.abs(lr.coef_[0]), index=X.columns)


def score_random_forest(X, y, random_state=SEED):
    return pd.Series(_rf(random_state).fit(X, y).feature_importances_, index=X.columns)


def score_shap(X, y, random_state=SEED, max_rows=400):
    import shap

    rf = _rf(random_state, n_estimators=200).fit(X, y)
    sample = X.sample(min(max_rows, len(X)), random_state=random_state)
    sv = shap.TreeExplainer(rf).shap_values(sample, check_additivity=False)
    sv = sv[..., 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)


SCORERS = {
    "ttest": score_ttest,
    "mutual_info": score_mutual_info,
    "l1_logistic": score_l1_logistic,
    "random_forest": score_random_forest,
    "shap": score_shap,
}


# ---------------------------------------------------------------- consensus
def consensus_ranking(X, y, methods=FAST_METHODS, random_state=SEED) -> pd.DataFrame:
    """One row per sensor: score + rank per method, mean rank and final consensus rank."""
    out = pd.DataFrame(index=X.columns)
    for m in methods:
        s = SCORERS[m](X, y, random_state=random_state)
        out[f"score_{m}"] = s
        out[f"rank_{m}"] = s.rank(ascending=False, method="average")
    out["mean_rank"] = out[[f"rank_{m}" for m in methods]].mean(axis=1)
    out["consensus_rank"] = out["mean_rank"].rank(method="first").astype(int)
    out.index.name = "sensor"
    return out.sort_values("consensus_rank")


class ConsensusSelector(BaseEstimator, TransformerMixin):
    """Keep the top-k sensors of the consensus ranking (fit on training data only)."""

    def __init__(self, k=20, methods=FAST_METHODS, random_state=SEED):
        self.k = k
        self.methods = methods
        self.random_state = random_state

    def fit(self, X, y):
        self.ranking_ = consensus_ranking(X, y, self.methods, self.random_state)
        self.selected_ = self.ranking_.index[: self.k].tolist()
        return self

    def transform(self, X):
        return X[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


def stability_selection(X, y, k=20, n_rounds=30, sample_frac=0.8, methods=FAST_METHODS,
                        random_state=SEED) -> pd.Series:
    """Share of stratified subsamples in which each sensor lands in the consensus top-k."""
    rng = np.random.default_rng(random_state)
    counts = pd.Series(0, index=X.columns, dtype=float)
    pos, neg = np.flatnonzero(y.to_numpy() == 1), np.flatnonzero(y.to_numpy() == 0)
    for r in range(n_rounds):
        idx = np.concatenate([rng.choice(pos, int(len(pos) * sample_frac), replace=False),
                              rng.choice(neg, int(len(neg) * sample_frac), replace=False)])
        rank = consensus_ranking(X.iloc[idx], y.iloc[idx], methods, random_state + r)
        counts[rank.index[:k]] += 1
    return (counts / n_rounds).rename("stability")


# ---------------------------------------------------------------- nested evaluation
MODELS = {
    "logistic": lambda rs: LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000),
    "random_forest": lambda rs: _rf(rs, n_estimators=200),
}


def best_threshold(y, p):
    """Threshold that minimises the balanced error rate (BER)."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.99, 50)))
    bers = [1 - balanced_accuracy_score(y, p >= t) for t in grid]
    return grid[int(np.argmin(bers))]


def evaluate_k(X, y, ks, methods=FAST_METHODS, models=("logistic", "random_forest"),
               n_splits=5, n_repeats=1, random_state=SEED) -> pd.DataFrame:
    """Nested CV: in every fold rank sensors on the training part, then score top-k on the held-out part.
    The decision threshold is tuned on inner out-of-fold predictions of the training part only."""
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        Xtr, ytr, Xva, yva = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]
        ranking = consensus_ranking(Xtr, ytr, methods, random_state + fold)
        for k in ks:
            cols = ranking.index[:k]
            for name in models:
                model = MODELS[name](random_state)
                inner = StratifiedKFold(3, shuffle=True, random_state=random_state)
                p_inner = cross_val_predict(model, Xtr[cols], ytr, cv=inner, method="predict_proba")[:, 1]
                thr = best_threshold(ytr, p_inner)
                p = model.fit(Xtr[cols], ytr).predict_proba(Xva[cols])[:, 1]
                pred = p >= thr
                rows.append({
                    "k": k, "model": name, "fold": fold,
                    "ber": 1 - balanced_accuracy_score(yva, pred),
                    "recall_fail": recall_score(yva, pred, zero_division=0),
                    "pr_auc": average_precision_score(yva, p),
                    "roc_auc": roc_auc_score(yva, p),
                })
    return pd.DataFrame(rows)


def choose_k(results: pd.DataFrame, metric="ber"):
    """Smallest k whose mean BER is within one standard error of the best (1-SE rule)."""
    g = results.groupby(["model", "k"])[metric].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    best_model, best_k = g["mean"].idxmin()
    limit = g.loc[(best_model, best_k), "mean"] + g.loc[(best_model, best_k), "se"]
    ok = g.loc[best_model]
    return best_model, int(ok.index[ok["mean"] <= limit].min()), g


# ---------------------------------------------------------------- script
def main(ks=(5, 10, 15, 20, 30, 40, 60, 100, 272), stability_k=20):
    train = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    sensors = [c for c in train.columns if c.startswith("sensor_")]
    X, y = train[sensors], train["fail"]

    print("1/3 consensus ranking (5 methods) ...")
    ranking = consensus_ranking(X, y, methods=ALL_METHODS)

    print("2/3 stability selection ...")
    ranking["stability"] = stability_selection(X, y, k=stability_k)

    print("3/3 nested CV over k ...")
    results = evaluate_k(X, y, ks=list(ks), n_repeats=2)
    best_model, k, summary = choose_k(results)

    ranking["selected"] = ranking["consensus_rank"] <= k
    ranking.to_csv(PROCESSED_DIR / "feature_ranking.csv")
    results.to_csv(PROCESSED_DIR / "feature_k_evaluation.csv", index=False)
    selected = ranking.index[:k].tolist()
    with open(MODELS_DIR / "selected_features.json", "w") as f:
        json.dump({"k": k, "chosen_with": best_model, "rule": "1-SE on nested-CV BER",
                   "features": selected}, f, indent=2)

    print(summary["mean"].unstack(0).round(3).to_string())
    print(f"\nChosen: k={k} (best model in CV: {best_model})")
    print(ranking.head(15)[["consensus_rank", "mean_rank", "stability"]].round(2).to_string())
    return ranking, results


if __name__ == "__main__":
    main()
