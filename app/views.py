"""Dashboard pages. Each page is a function; `dashboard.py` wires them into navigation."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from app.data import (fail_signature, load_json, load_model, load_table, load_units, score_raw,
                      sensor_columns, unit_contributions, weekly)
from src.spc import PercentileLimits, western_electric

CHART = {"width": "stretch", "config": {"displaylogo": False}}


# ---------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner="Loading production data ...")
def units_df():
    return load_units()


@st.cache_data
def table(name, index_col=None):
    return load_table(name, index_col=index_col)


@st.cache_data
def card():
    return load_json("model_card.json")


@st.cache_resource
def model(name):
    return load_model(name)


def test_start():
    u = units_df()
    return u.loc[u.period == "Test", "timestamp"].min()


# ---------------------------------------------------------------- shared filter
def filtered_units():
    """Global filter (sidebar): period + date range. Every chart on a page uses the same slice."""
    u = units_df()
    period = st.session_state.get("period", "All")
    if period != "All":
        u = u[u.period == period]
    rng = st.session_state.get("dates")
    if rng and len(rng) == 2:
        u = u[(u.timestamp.dt.date >= rng[0]) & (u.timestamp.dt.date <= rng[1])]
    return u


def sidebar_filters():
    u = units_df()
    st.sidebar.radio("Period", ["All", "Train", "Test"], key="period", horizontal=True,
                     help="Train = 19 Jul – 29 Sep 2008 (model fitting). Test = 29 Sep – 17 Oct 2008 (unseen).")
    lo, hi = u.timestamp.min().date(), u.timestamp.max().date()
    st.sidebar.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key="dates")
    f = filtered_units()
    st.sidebar.caption(f"**{len(f):,}** units in view · {int(f.fail.sum())} failures")


def table_view(df, label="View data table"):
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True)


def empty_guard(u):
    if u.empty:
        st.info("No units in the selected filter. Widen the period or date range in the sidebar.")
        st.stop()


# ---------------------------------------------------------------- pages
def overview():
    st.title("Semiconductor Quality Intelligence")
    st.caption("SECOM fab data · 1,567 production units · 590 sensors · Jul–Oct 2008")
    u = filtered_units()
    empty_guard(u)

    c = st.columns(5)
    c[0].metric("Units tested", f"{len(u):,}")
    c[1].metric("Yield", f"{1 - u.fail.mean():.1%}", help="Share of units that passed in-house line test")
    c[2].metric("Failures", f"{int(u.fail.sum())}")
    c[3].metric("High-risk units", f"{int((u.risk_band == 'High').sum())}",
                help="Fail model probability above its decision threshold")
    c[4].metric("⚠ Sensor/tool alarms", f"{int(u.mspc_alarm.sum())}",
                help="Units beyond the multivariate SPC limit (T² or SPE)")

    w = weekly(u)
    st.plotly_chart(charts.weekly_fail_rate(w, u.fail.mean(), test_start()), **CHART)
    st.plotly_chart(charts.weekly_volume(w, test_start()), **CHART)
    table_view(w.assign(fail_rate=lambda d: (d.fail_rate * 100).round(1)), "Weekly table")

    st.subheader("What the platform found")
    st.markdown("""
| Area | Finding |
|---|---|
| **Yield trend** | Fail rate fell from ~22% (July) to ~3% (September), with a small spike in early October |
| **Root cause** | 7 core sensors are stable across methods; `sensor_059` and `sensor_129` rank top in 100% of subsamples |
| **Prediction** | The fail signature **drifts**: `sensor_059` separates failures at 1.8σ in September and not at all in October. The model catches 9/24 test failures, which makes it a risk-ranking aid, not an auto-scrap rule |
| **Equipment health** | Multivariate SPC flags sensor/tool events on *passing* units: `sensor_140` ≈ 9999 error codes, a `sensor_007` burst |
""")


def spc_monitor():
    st.title("SPC Monitor")
    st.caption("Individuals chart with percentile control limits learned on Pass units of the train period. "
               "Western Electric rules: R1/R2 raise an alarm, R3/R4 a shift/trend warning.")
    u_all = units_df()
    limits = table("spc_limits.csv")
    monitored = limits.sort_values("rank")["sensor"].tolist()
    ranking = table("feature_ranking.csv", index_col="sensor")
    others = [s for s in ranking.index if s not in monitored]
    options = monitored + ["sensor_140"] + [s for s in others if s != "sensor_140"]

    sensor = st.selectbox("Sensor", options, index=0,
                          format_func=lambda s: f"{s}  (monitored)" if s in monitored else s,
                          help="The 10 monitored sensors come first; sensor_140 is the Step 3 error-code case study.")
    baseline = u_all[(u_all.period == "Train") & (u_all.fail == 0)]
    lim = PercentileLimits.fit(baseline[sensor], sensor)
    flags_all = western_electric(u_all[sensor], lim)

    u = filtered_units()
    empty_guard(u)
    flags = flags_all.loc[u.index]
    c = st.columns(4)
    c[0].metric("Readings", f"{u[sensor].notna().sum():,}")
    c[1].metric("⚠ Alarms (R1/R2)", int(flags.alarm.sum()))
    c[2].metric("Warnings (R3/R4)", int((flags.warning & ~flags.alarm).sum()))
    c[3].metric("Fails among alarmed", f"{int(u.loc[flags.alarm, 'fail'].sum())} / {int(flags.alarm.sum())}")
    st.plotly_chart(charts.control_chart(u, sensor, lim, flags, test_start()), **CHART)

    v = u.loc[flags.any_violation, ["unit_id", "timestamp", "status", sensor]].copy()
    v["rules"] = [", ".join(r.upper() for r in ["rule1", "rule2", "rule3", "rule4"] if flags.at[i, r]) for i in v.index]
    v["severity"] = np.where(flags.loc[v.index, "alarm"], "⚠ alarm", "warning")
    table_view(v.sort_values("timestamp", ascending=False), f"Violations ({len(v)})")
    table_view(limits, "Control limits of all monitored sensors")


def root_cause():
    st.title("Root Cause")
    st.caption("Consensus of 5 feature-ranking methods (t-test, mutual information, L1 logistic, random forest, SHAP) "
               "on the train period, with bootstrap stability.")
    ranking = table("feature_ranking.csv", index_col="sensor")
    left, right = st.columns([1, 1])
    with left:
        k = st.slider("Sensors shown", 10, 30, 20, step=5)
        st.plotly_chart(charts.importance_bar(ranking, k), **CHART)
    with right:
        sensor = st.selectbox("Inspect sensor", ranking.index[:30].tolist())
        u = filtered_units()
        empty_guard(u)
        st.plotly_chart(charts.pass_fail_box(u, sensor), **CHART)
        r = ranking.loc[sensor]
        st.markdown(f"Consensus rank **#{int(r.consensus_rank)}** · stability **{r.stability:.0%}** · "
                    f"selected for the model: **{'yes' if r.selected else 'no'}**")

    u_all = units_df()
    periods = {"Jul–Aug": (u_all.period == "Train") & (u_all.timestamp < "2008-09-01"),
               "Sep": (u_all.period == "Train") & (u_all.timestamp >= "2008-09-01"),
               "Oct (test)": u_all.period == "Test"}
    sig = fail_signature(u_all, ranking.index[:10].tolist(), periods)
    st.plotly_chart(charts.signature_heatmap(sig), **CHART)
    st.caption("Red = failed units read higher than passing ones; blue = lower. A sensor whose colour fades or flips "
               "between periods has lost its predictive power: concept drift.")
    cols = ["consensus_rank", "mean_rank", "stability", "selected"] + [c for c in ranking.columns if c.startswith("rank_")]
    table_view(ranking[cols].reset_index().round(2), "Full ranking (272 sensors)")


def fail_risk():
    st.title("Fail Risk")
    mc = card()
    t, ci = mc["test"], mc["test_ci95"]
    st.caption(f"Model: **{mc['model'].replace('_', ' ')}** on {len(mc['features'])} sensors, chosen by forward "
               f"(time-aware) validation. Threshold {mc['threshold']:.3f}. Intended use: {mc['intended_use']}.")
    c = st.columns(4)
    c[0].metric("Test BER", f"{t['ber']:.1%}", help=f"95% CI {ci['ber'][0]:.1%} – {ci['ber'][1]:.1%}. 50% = coin flip")
    c[1].metric("Failures caught", f"{t['tp']} / {t['tp'] + t['fn']}")
    c[2].metric("Units flagged", f"{t['flag_rate']:.1%}")
    c[3].metric("Test ROC-AUC", f"{t['roc_auc']:.2f}", help=f"95% CI {ci['roc_auc'][0]:.2f} – {ci['roc_auc'][1]:.2f}")
    st.warning("Performance is modest because the failure signature drifts between months (see Root Cause). "
               "Use the score to prioritise inspection, and retrain as new failures are labelled.", icon="⚠️")

    u = filtered_units()
    empty_guard(u)
    st.caption("Train-period probabilities are out-of-fold; test-period probabilities come from the frozen model.")
    left, right = st.columns(2)
    left.plotly_chart(charts.risk_band_bar(u), **CHART)
    if u.fail.nunique() == 2:
        right.plotly_chart(charts.gain_chart(u), **CHART)

    st.subheader("Highest-risk units")
    top = u.sort_values("fail_prob", ascending=False)[["unit_id", "timestamp", "period", "fail_prob", "risk_band", "status"]]
    st.dataframe(top.head(25), width="stretch", hide_index=True,
                 column_config={"fail_prob": st.column_config.ProgressColumn("fail probability", min_value=0, max_value=1, format="%.2f")})

    st.subheader("Explain one unit")
    unit = st.selectbox("Unit", top.unit_id.tolist())
    row = units_df().loc[lambda d: d.unit_id == unit]
    fm = model("fail_model.joblib")
    contrib = unit_contributions(fm, row[sensor_columns(row)])
    r = row.iloc[0]
    st.markdown(f"**{unit}** · {r.timestamp:%d %b %Y %H:%M} · fail probability **{r.fail_prob:.2f}** "
                f"({r.risk_band} risk) · actual result: **{r.status}**")
    if not contrib.empty:
        st.plotly_chart(charts.contribution_bar(contrib, "Sensor contributions to the log-odds of failure"), **CHART)

    st.subheader("Score new units")
    st.caption("Upload a CSV with the 590 raw sensor columns `sensor_000` … `sensor_589` (optional `unit_id`). "
               "The saved pipeline cleans, selects and scores them; the anomaly detector checks sensor health.")
    sample = units_df().loc[units_df().period == "Test", ["unit_id"] + sensor_columns(units_df())].head(5)
    st.download_button("Download a sample file (5 test units)", sample.to_csv(index=False), "sample_units.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()))
            scored = score_raw(raw, fm, model("anomaly_detector.joblib"), mc["threshold"])
            st.dataframe(scored, width="stretch", hide_index=True)
        except ValueError as e:
            st.error(f"Could not score the file: {e}")


def equipment_health():
    st.title("Equipment & Sensor Health")
    st.caption("Multivariate SPC (PCA Hotelling T² + SPE) fitted on Pass units of the train period, 99% out-of-fold "
               "limits. It flags units whose sensor pattern is abnormal and names the sensors behind it.")
    u = filtered_units()
    empty_guard(u)
    alarms = u[u.mspc_alarm == True]  # noqa: E712 (column may hold numpy bools)
    c = st.columns(4)
    c[0].metric("⚠ Units in alarm", len(alarms))
    c[1].metric("Alarm rate", f"{len(alarms) / len(u):.1%}")
    c[2].metric("Failures among alarms", int(alarms.fail.sum()))
    c[3].metric("Isolation Forest flags", int(u.iso_alarm.sum()))
    st.plotly_chart(charts.anomaly_timeline(u, test_start()), **CHART)
    st.caption("Train-period Pass units form the baseline itself, so they sit mostly below the limit by construction.")

    if alarms.empty:
        st.info("No alarms in the selected range.")
        return
    ev = alarms.assign(main_sensor=alarms.top_sensors.str.split(", ").str[0]).groupby("main_sensor").agg(
        units=("unit_id", "count"), first=("timestamp", "min"), last=("timestamp", "max"),
        fails=("fail", "sum"), max_severity=("anomaly_severity", "max")).sort_values("units", ascending=False)
    st.subheader("Events grouped by main driver sensor")
    st.dataframe(ev.reset_index(), width="stretch", hide_index=True,
                 column_config={"max_severity": st.column_config.NumberColumn("max × limit", format="%.1f")})

    st.subheader("Why was this unit flagged?")
    unit = st.selectbox("Alarmed unit", alarms.sort_values("anomaly_severity", ascending=False).unit_id.tolist())
    row = units_df().loc[lambda d: d.unit_id == unit]
    det = model("anomaly_detector.joblib")
    clean = model("fail_model.joblib").named_steps["clean"].transform(row[sensor_columns(row)])
    r = row.iloc[0]
    contrib = (det.spe_contributions(clean) if r.driver == "SPE" else det.t2_contributions(clean)).iloc[0]
    share = contrib / contrib.sum() * 100
    st.markdown(f"**{unit}** · {r.timestamp:%d %b %H:%M} · {r.anomaly_severity:,.1f} × limit ({r.driver}) · "
                f"final test: **{r.status}**")
    st.plotly_chart(charts.contribution_bar(share, f"{r.driver} contribution by sensor", diverging=False), **CHART)


def about():
    st.title("About this project")
    rep = load_json_processed("cleaning_report.json")
    st.markdown(f"""
An end-to-end quality-intelligence platform built on the public **UCI SECOM** dataset: sensor data from a
semiconductor fab, with the pass/fail result of in-house line testing for each unit.

**Pipeline** (each step is a tested Python module in `src/` with a notebook in `notebooks/`):

| Step | Module | Output |
|---|---|---|
| 1 Data understanding | `data_loader.py` | merged unit table |
| 2 Cleaning | `preprocessing.py` | {rep['raw_sensors']} → {rep['final_sensors']} sensors, chronological train/test split, fitted on train only |
| 3 SPC | `spc.py` | percentile control limits + Western Electric rules |
| 4 Feature selection | `feature_selection.py` | consensus of 5 methods, stability selection, nested CV |
| 5 Fail model | `model.py` | forward-validated logistic model + model card |
| 6 Anomaly detection | `anomaly.py` | PCA T²/SPE + Isolation Forest with contributions |
| 7 Dashboard | `app/` | this app |

**Data split:** train {rep['train']['from'][:10]} → {rep['train']['to'][:10]} ({rep['train']['units']} units,
{rep['train']['fails']} fails) · test {rep['test']['from'][:10]} → {rep['test']['to'][:10]}
({rep['test']['units']} units, {rep['test']['fails']} fails).

**Honest limits:** the test period has only 24 failures, so every test metric has a wide confidence interval, and the
failure mechanism changes over time. The dashboard is meant to support engineers, not to make decisions automatically.

Data: McCann & Johnston, *SECOM* (UCI Machine Learning Repository, 2008).
""")


def load_json_processed(name):
    import json
    from app.data import PROCESSED
    with open(PROCESSED / name) as f:
        return json.load(f)
