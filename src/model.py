"""
Fail-prediction model for SECOM production units.

Protocol (v2, time-aware):
  1. Candidates are compared on the TRAIN period with FORWARD validation: fit on the
     first 75% of the train period (Jul-Sep), score the last 25% (September).
     Stratified CV is also reported, only to show how much it over-states performance
     when the process drifts.
  2. The decision threshold is tuned on out-of-fold probabilities of the full train period (min BER).
  3. The chosen model is refit on the full train period and scored on the test period
     (29 Sep -> 17 Oct 2008) with bootstrap confidence intervals.
  4. Deployment simulation: weekly walk-forward retraining over the test period.
  5. The final artefact is a single pipeline: raw 590 sensors -> cleaning -> 30 sensors -> model,
     so the dashboard can score a new unit straight from raw readings.

Note: a first run (v1) selected the model with stratified CV only; its test result is
reported in notebooks/05_fail_prediction.ipynb for transparency.

Run as a script:
    python -m src.model
"""
import json

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_loader import PROCESSED_DIR, load_secom
from src.preprocessing import MODELS_DIR, build_preprocessor, time_split

SEED = 42

CANDIDATES = {
    "logistic": lambda: LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000),
    "logistic_smote": lambda: ImbPipeline([
        ("smote", SMOTE(k_neighbors=5, random_state=SEED)),
        ("model", LogisticRegression(C=0.05, max_iter=5000)),
    ]),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=3,
        n_jobs=-1, random_state=SEED),
    "balanced_rf": lambda: BalancedRandomForestClassifier(
        n_estimators=500, sampling_strategy="all", replacement=True, bootstrap=False,
        min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "hist_gb": lambda: HistGradientBoostingClassifier(
        class_weight="balanced", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED),
}


def build_candidate(name):
    return CANDIDATES[name]()


class ColumnPicker(BaseEstimator, TransformerMixin):
    """Select (and order) a fixed list of columns from a DataFrame."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


# ---------------------------------------------------------------- metrics
def classification_metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "ber": 1 - (recall + spec) / 2,
        "recall_fail": recall,
        "specificity": spec,
        "precision_fail": tp / (tp + fp) if tp + fp else 0.0,
        "flag_rate": float(pred.mean()),
        "pr_auc": average_precision_score(y, p) if both else np.nan,
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def tune_threshold(y, p, max_candidates=500):
    """Threshold on predicted probability that minimises the balanced error rate."""
    y, p = np.asarray(y), np.asarray(p)
    cand = np.unique(p)
    if len(cand) > max_candidates:
        cand = np.unique(np.quantile(p, np.linspace(0, 1, max_candidates)))
    pos, neg = (y == 1).sum(), (y == 0).sum()
    best_t, best_ber = cand[0], np.inf
    for t in cand:
        pred = p >= t
        ber = 1 - ((pred & (y == 1)).sum() / pos + (~pred & (y == 0)).sum() / neg) / 2
        if ber < best_ber:
            best_t, best_ber = t, ber
    return float(best_t)


def risk_band(p, threshold):
    """High = model flags the unit; Medium = at least half the threshold; Low = rest."""
    p = np.asarray(p)
    return np.select([p >= threshold, p >= threshold / 2], ["High", "Medium"], "Low")


def bootstrap_ci(y, p, metric_fn, n_boot=1000, alpha=0.05, random_state=SEED):
    """Percentile bootstrap CI of metric_fn(y, p) over resampled units."""
    rng = np.random.default_rng(random_state)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metric_fn(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ---------------------------------------------------------------- CV
def oof_probabilities(model, X, y, n_splits=5, n_repeats=2, random_state=SEED):
    """Out-of-fold fail probability for every unit, averaged over repeats."""
    p = np.zeros(len(y))
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
        for tr, va in cv.split(X, y):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            p[va] += m.predict_proba(X.iloc[va])[:, 1]
    return p / n_repeats


def compare_candidates(X, y, names=tuple(CANDIDATES)):
    """Stratified (time-mixing) CV. Kept for contrast: it over-states performance on drifting data."""
    rows, oof = [], {}
    for name in names:
        p = oof_probabilities(build_candidate(name), X, y)
        t = tune_threshold(y, p)
        oof[name] = p
        rows.append({"model": name, **classification_metrics(y, p, t)})
    return pd.DataFrame(rows).sort_values("ber").reset_index(drop=True), oof


def forward_validation(X, y, ts, names=tuple(CANDIDATES), valid_frac=0.25):
    """Time-aware model selection: fit on the earlier part, score the later part.
    The threshold comes from out-of-fold predictions on the earlier part only."""
    order = np.argsort(ts.to_numpy(), kind="stable")
    X, y, ts = X.iloc[order], y.iloc[order], ts.iloc[order]
    cut = ts.iloc[int(len(ts) * (1 - valid_frac))]
    early, late = (ts < cut).to_numpy(), (ts >= cut).to_numpy()
    rows = []
    for name in names:
        model = build_candidate(name)
        t = tune_threshold(y[early], oof_probabilities(model, X[early], y[early], n_repeats=1))
        p = clone(model).fit(X[early], y[early]).predict_proba(X[late])[:, 1]
        rows.append({"model": name, **classification_metrics(y[late], p, t)})
    return pd.DataFrame(rows).sort_values("ber").reset_index(drop=True), cut


def walk_forward(name, X, y, ts, start, step_days=7):
    """Deployment simulation: every `step_days`, retrain on ALL history before the window
    (threshold re-tuned on that history) and score the units of the next window."""
    ts = pd.Series(pd.to_datetime(ts.to_numpy()), index=X.index)
    out, a = [], pd.Timestamp(start)
    while a <= ts.max():
        b = a + pd.Timedelta(days=step_days)
        hist, nxt = (ts < a).to_numpy(), ((ts >= a) & (ts < b)).to_numpy()
        if nxt.any():
            model = build_candidate(name)
            t = tune_threshold(y[hist], oof_probabilities(model, X[hist], y[hist], n_repeats=1))
            p = clone(model).fit(X[hist], y[hist]).predict_proba(X[nxt])[:, 1]
            out.append(pd.DataFrame({
                "timestamp": ts[nxt], "fail": y[nxt], "fail_prob": p, "threshold": t,
                "predicted_fail": (p >= t).astype(int), "train_until": a,
            }, index=X.index[nxt]))
        a = b
    return pd.concat(out)


# ---------------------------------------------------------------- script
def main():
    train = pd.read_csv(PROCESSED_DIR / "train_clean.csv", parse_dates=["timestamp"])
    test = pd.read_csv(PROCESSED_DIR / "test_clean.csv", parse_dates=["timestamp"])
    features = json.load(open(MODELS_DIR / "selected_features.json"))["features"]
    Xtr, ytr, Xte, yte = train[features], train["fail"], test[features], test["fail"]

    print("1/5 stratified CV on the train period (for contrast) ...")
    cv_table, oof = compare_candidates(Xtr, ytr)

    print("2/5 forward validation on the train period (model selection) ...")
    fwd_table, cut = forward_validation(Xtr, ytr, train["timestamp"])
    best = fwd_table.loc[0, "model"]
    threshold = float(cv_table.set_index("model").loc[best, "threshold"])
    both = cv_table.merge(fwd_table, on="model", suffixes=("_stratified", "_forward"))
    print(both[["model", "ber_stratified", "ber_forward", "roc_auc_stratified", "roc_auc_forward"]]
          .sort_values("ber_forward").round(3).to_string(index=False))
    print(f"-> selected by forward validation: {best} (threshold {threshold:.3f}, validation from {cut:%d %b})")

    print("3/5 refit on full train, score the test period ...")
    model = build_candidate(best).fit(Xtr, ytr)
    p_test = model.predict_proba(Xte)[:, 1]
    test_m = classification_metrics(yte, p_test, threshold)
    ci = {
        "ber": bootstrap_ci(yte, p_test, lambda a, b: classification_metrics(a, b, threshold)["ber"]),
        "recall_fail": bootstrap_ci(yte, p_test, lambda a, b: classification_metrics(a, b, threshold)["recall_fail"]),
        "pr_auc": bootstrap_ci(yte, p_test, average_precision_score),
        "roc_auc": bootstrap_ci(yte, p_test, roc_auc_score),
    }

    print("4/5 reference points + weekly walk-forward retraining ...")
    all_cols = [c for c in train.columns if c.startswith("sensor_")]
    p_all = build_candidate(best).fit(train[all_cols], ytr).predict_proba(test[all_cols])[:, 1]
    t_all = tune_threshold(ytr, oof_probabilities(build_candidate(best), train[all_cols], ytr))
    spc = pd.read_csv(PROCESSED_DIR / "spc_unit_alarms.csv")
    spc_test = spc[spc.period == "test"].set_index("unit_id").loc[test["unit_id"], "spc_alarm"].to_numpy()
    rows = [{"approach": f"{best} (30 sensors) - FINAL", **test_m}]
    for name in CANDIDATES:
        if name != best:
            t_o = float(cv_table.set_index("model").loc[name, "threshold"])
            p_o = build_candidate(name).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            rows.append({"approach": f"{name} (30 sensors)", **classification_metrics(yte, p_o, t_o)})
    rows.append({"approach": f"{best} (all 272 sensors)", **classification_metrics(yte, p_all, t_all)})

    both_periods = pd.concat([train, test], ignore_index=True)
    wf = walk_forward(best, both_periods[features], both_periods["fail"], both_periods["timestamp"],
                      start=test["timestamp"].min().normalize())
    wf_m = classification_metrics(wf["fail"], wf["predicted_fail"].astype(float), 0.5)
    wf_m["roc_auc"] = roc_auc_score(wf["fail"], wf["fail_prob"])
    wf_m["pr_auc"] = average_precision_score(wf["fail"], wf["fail_prob"])
    rows.append({"approach": f"{best} weekly walk-forward retraining", **wf_m})
    rows.append({"approach": "SPC alarm rule (Step 3)", **classification_metrics(yte, spc_test.astype(float), 0.5)})
    rows.append({"approach": "Always predict Pass", **classification_metrics(yte, np.zeros(len(yte)), 0.5)})
    baselines = pd.DataFrame(rows)
    print(baselines[["approach", "ber", "recall_fail", "specificity", "precision_fail", "roc_auc"]]
          .round(3).to_string(index=False))

    print("5/5 saving final raw-to-prediction pipeline ...")
    raw = load_secom()
    raw_train, _ = time_split(raw)
    raw_sensors = [c for c in raw.columns if c.startswith("sensor_")]
    final = Pipeline([
        ("clean", build_preprocessor()),
        ("select", ColumnPicker(features)),
        ("model", build_candidate(best)),
    ]).fit(raw_train[raw_sensors], raw_train["fail"])
    joblib.dump(final, MODELS_DIR / "fail_model.joblib")

    card = {
        "model": best,
        "selection": "forward validation inside train period (last 25% held out)",
        "features": features,
        "threshold": threshold,
        "train_period": [str(train.timestamp.min()), str(train.timestamp.max())],
        "test_period": [str(test.timestamp.min()), str(test.timestamp.max())],
        "stratified_cv_train": cv_table.set_index("model").loc[best].drop("threshold").to_dict(),
        "forward_validation_train": fwd_table.set_index("model").loc[best].drop("threshold").to_dict(),
        "test": test_m,
        "test_ci95": ci,
        "walk_forward_test": wf_m,
        "input": "raw 590 SECOM sensor columns named sensor_000..sensor_589",
        "intended_use": "decision support / risk ranking with regular retraining; not automatic scrap",
    }
    with open(MODELS_DIR / "model_card.json", "w") as f:
        json.dump(card, f, indent=2, default=float)

    preds = pd.concat([
        train[["unit_id", "timestamp", "fail"]].assign(period="train", fail_prob=oof[best], source="out-of-fold"),
        test[["unit_id", "timestamp", "fail"]].assign(period="test", fail_prob=p_test, source="holdout"),
    ], ignore_index=True)
    preds["predicted_fail"] = (preds["fail_prob"] >= threshold).astype(int)
    preds["risk_band"] = risk_band(preds["fail_prob"], threshold)
    wf_prob = pd.Series(wf["fail_prob"].to_numpy(), index=both_periods.loc[wf.index, "unit_id"])
    preds["walk_forward_prob"] = preds["unit_id"].map(wf_prob)
    preds.to_csv(PROCESSED_DIR / "predictions.csv", index=False)
    both.to_csv(PROCESSED_DIR / "model_cv_comparison.csv", index=False)
    baselines.to_csv(PROCESSED_DIR / "model_test_comparison.csv", index=False)
    wf.assign(unit_id=both_periods.loc[wf.index, "unit_id"]).to_csv(PROCESSED_DIR / "walk_forward.csv", index=False)

    print(f"\nTEST  BER {test_m['ber']:.3f} (95% CI {ci['ber'][0]:.3f}-{ci['ber'][1]:.3f}) | "
          f"recall {test_m['recall_fail']:.2f} ({test_m['tp']}/{test_m['tp'] + test_m['fn']} fails caught) | "
          f"ROC-AUC {test_m['roc_auc']:.3f} (CI {ci['roc_auc'][0]:.2f}-{ci['roc_auc'][1]:.2f})")
    print(f"WALK-FORWARD  BER {wf_m['ber']:.3f} | recall {wf_m['recall_fail']:.2f} | ROC-AUC {wf_m['roc_auc']:.3f}")
    return card


if __name__ == "__main__":
    # import via the package so pickled classes (ColumnPicker) reference `src.model`, not `__main__`
    from src.model import main as _main
    _main()
