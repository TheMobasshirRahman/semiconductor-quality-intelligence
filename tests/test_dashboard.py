"""Smoke tests: every dashboard page renders without exceptions on the real pipeline outputs."""
import pandas as pd
import plotly.graph_objects as go
import pytest

from app import charts
from app.data import load_units, missing_artifacts, weekly

pytestmark = pytest.mark.skipif(bool(missing_artifacts()), reason="pipeline outputs not built")

PAGES = ["overview", "spc_monitor", "root_cause", "fail_risk", "equipment_health", "about"]


def _page_script(page_name):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    from app import views

    views.sidebar_filters()
    getattr(views, page_name)()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_page_script, args=(page,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_test_period_filter_changes_kpis():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_page_script, args=("overview",), default_timeout=120)
    at.run()
    all_units = at.metric[0].value
    at.sidebar.radio[0].set_value("Test").run()
    assert not at.exception
    assert at.metric[0].value == "392" and all_units == "1,567"


def test_unit_table_joins_every_step():
    u = load_units()
    assert len(u) == 1567 and u.unit_id.is_unique
    for col in ["fail_prob", "risk_band", "mspc_alarm", "spc_alarm", "top_sensors", "period"]:
        assert u[col].notna().all(), col
    assert set(u.period) == {"Train", "Test"}


def test_charts_use_a_single_y_axis():
    u = load_units()
    figs = [charts.weekly_fail_rate(weekly(u), u.fail.mean()), charts.risk_band_bar(u),
            charts.gain_chart(u), charts.anomaly_timeline(u)]
    for fig in figs:
        assert isinstance(fig, go.Figure)
        assert "yaxis2" not in fig.layout.to_plotly_json()


def test_score_raw_matches_saved_predictions():
    from app.data import load_json, load_model, score_raw, sensor_columns

    u = load_units()
    test = u[u.period == "Test"].head(20)
    out = score_raw(test[["unit_id"] + sensor_columns(u)], load_model("fail_model.joblib"),
                    load_model("anomaly_detector.joblib"), load_json("model_card.json")["threshold"])
    pd.testing.assert_series_equal(out["fail_prob"].reset_index(drop=True),
                                   test["fail_prob"].round(4).reset_index(drop=True), check_names=False, atol=1e-4)
