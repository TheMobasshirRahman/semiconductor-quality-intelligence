"""
Unsupervised anomaly detection: flag units that do not look like normal production,
without needing failure labels. This targets exactly what the supervised model cannot
learn: NEW failure modes and process / sensor drift.

Multivariate SPC (MSPC) with PCA - the industry standard for many correlated sensors:
    Hotelling T²  distance INSIDE the normal PCA plane  -> "normal pattern, but extreme"
    SPE (Q)       distance OFF the normal PCA plane     -> "sensor relationships broken"
plus an Isolation Forest as a non-linear second opinion.

- Baseline = Pass units of the TRAIN period (normal operation).
- Control limits = (1 - alpha) quantile of OUT-OF-FOLD baseline scores, so the limits are
  not flattered by scoring the same units the model was fitted on. Empirical quantiles
  (not F / chi² formulas) because the sensors are far from normal (see Step 3).
- Contributions tell the engineer WHICH sensors drive an alarm.

Run as a script:
    python -m src.anomaly
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROCESSED_DIR
from src.preprocessing import MODELS_DIR

SEED = 42


class AnomalyDetector(BaseEstimator):
    def __init__(self, variance=0.9, alpha=0.01, n_folds=5, n_estimators=300, top_k=3,
                 random_state=SEED):
        self.variance = variance
        self.alpha = alpha
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.top_k = top_k
        self.random_state = random_state

    # ------------------------------------------------------------ fitting
    def _fit_core(self, X):
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        pca = PCA(n_components=self.variance, svd_solver="full", random_state=self.random_state).fit(Xs)
        iso = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state,
                              n_jobs=-1).fit(Xs)
        return scaler, pca, iso

    @staticmethod
    def _stats(core, X):
        scaler, pca, iso = core
        Xs = scaler.transform(X)
        T = pca.transform(Xs)
        resid = Xs - pca.inverse_transform(T)
        t2 = np.sum(T ** 2 / pca.explained_variance_, axis=1)
        spe = np.sum(resid ** 2, axis=1)
        iso_score = -iso.score_samples(Xs)
        return t2, spe, iso_score

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        X = X[self.columns_]
        oof = np.zeros((len(X), 3))
        cv = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for tr, va in cv.split(X):
            core = self._fit_core(X.iloc[tr])
            oof[va] = np.column_stack(self._stats(core, X.iloc[va]))
        q = 1 - self.alpha
        self.t2_limit_, self.spe_limit_, self.iso_limit_ = np.quantile(oof, q, axis=0)
        self.scaler_, self.pca_, self.iso_ = self._fit_core(X)
        self.n_components_ = int(self.pca_.n_components_)
        return self

    # ------------------------------------------------------------ scoring
    def _core(self):
        return self.scaler_, self.pca_, self.iso_

    def _t2(self, X):
        return self._stats(self._core(), X[self.columns_])[0]

    def _spe(self, X):
        return self._stats(self._core(), X[self.columns_])[1]

    def spe_contributions(self, X):
        """Squared residual per sensor: which sensors break the normal correlation structure."""
        Xs = self.scaler_.transform(X[self.columns_])
        resid = Xs - self.pca_.inverse_transform(self.pca_.transform(Xs))
        return pd.DataFrame(resid ** 2, index=X.index, columns=self.columns_)

    def t2_contributions(self, X):
        """Per-sensor contribution to T² (sum over components of t_a/λ_a · p_aj · x_j), negatives clipped."""
        Xs = self.scaler_.transform(X[self.columns_])
        T = self.pca_.transform(Xs)
        w = (T / self.pca_.explained_variance_) @ self.pca_.components_
        return pd.DataFrame(np.clip(w * Xs, 0, None), index=X.index, columns=self.columns_)

    def score(self, X):
        t2, spe, iso = self._stats(self._core(), X[self.columns_])
        out = pd.DataFrame({
            "t2": t2, "spe": spe, "iso_score": iso,
            "t2_ratio": t2 / self.t2_limit_, "spe_ratio": spe / self.spe_limit_,
            "iso_ratio": iso / self.iso_limit_,
        }, index=X.index)
        out["t2_alarm"] = out["t2_ratio"] > 1
        out["spe_alarm"] = out["spe_ratio"] > 1
        out["iso_alarm"] = out["iso_ratio"] > 1
        out["mspc_alarm"] = out["t2_alarm"] | out["spe_alarm"]
        out["n_flags"] = out[["t2_alarm", "spe_alarm", "iso_alarm"]].sum(axis=1)

        use_spe = (out["spe_ratio"] >= out["t2_ratio"]).to_numpy()
        spe_c, t2_c = self.spe_contributions(X).to_numpy(), self.t2_contributions(X).to_numpy()
        contrib = np.where(use_spe[:, None], spe_c, t2_c)
        top = np.argsort(-contrib, axis=1)[:, : self.top_k]
        names = np.asarray(self.columns_)
        out["driver"] = np.where(use_spe, "SPE", "T2")
        out["top_sensors"] = [", ".join(names[r]) for r in top]
        return out


# ---------------------------------------------------------------- script
def main():
    train = pd.read_csv(PROCESSED_DIR / "train_clean.csv", parse_dates=["timestamp"])
    test = pd.read_csv(PROCESSED_DIR / "test_clean.csv", parse_dates=["timestamp"])
    sensors = [c for c in train.columns if c.startswith("sensor_")]
    baseline = train[train["fail"] == 0]

    # variance=0.5 chosen by forward check inside the train period (Jul-Aug baseline -> Sep), see notebook 06
    det = AnomalyDetector(variance=0.5).fit(baseline[sensors])
    joblib.dump(det, MODELS_DIR / "anomaly_detector.joblib")

    units = pd.concat([train.assign(period="train"), test.assign(period="test")], ignore_index=True)
    scores = det.score(units[sensors])
    out = pd.concat([units[["unit_id", "timestamp", "period", "fail"]], scores], axis=1)
    out.to_csv(PROCESSED_DIR / "anomaly_scores.csv", index=False)

    print(f"PCA components: {det.n_components_} (explain {det.variance:.0%} of baseline variance) | "
          f"limits: T² {det.t2_limit_:.1f}, SPE {det.spe_limit_:.1f}, ISO {det.iso_limit_:.3f}")
    t = out[out.period == "test"]
    for flag in ["t2_alarm", "spe_alarm", "mspc_alarm", "iso_alarm"]:
        a = t[flag]
        print(f"test {flag:10s}: flags {a.mean():6.1%} of units | pass units {t.loc[t.fail == 0, flag].mean():6.1%} | "
              f"fails caught {int(t.loc[t.fail == 1, flag].sum())}/{int(t.fail.sum())} | "
              f"fail rate flagged {t.loc[a, 'fail'].mean() if a.any() else 0:.1%} vs not {t.loc[~a, 'fail'].mean():.1%}")
    from sklearn.metrics import roc_auc_score
    print("test ROC-AUC of raw scores:", {k: round(roc_auc_score(t.fail, t[k]), 3) for k in ["t2", "spe", "iso_score"]})
    return det, out


if __name__ == "__main__":
    # import via the package so the pickled detector references `src.anomaly`, not `__main__`
    from src.anomaly import main as _main
    _main()
