"""
Streamlit entry point.

    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import views  # noqa: E402
from app.data import missing_artifacts  # noqa: E402

st.set_page_config(page_title="Semiconductor Quality Intelligence", page_icon="🔬", layout="wide")

missing = missing_artifacts()
if missing:
    st.error("Some pipeline outputs are missing. Run these commands from the project root, then reload:")
    st.code("\n".join(dict.fromkeys(cmd for _, cmd in missing)))
    st.stop()

pages = [
    st.Page(views.overview, title="Overview", icon="📊", default=True),
    st.Page(views.spc_monitor, title="SPC Monitor", icon="📈", url_path="spc"),
    st.Page(views.root_cause, title="Failure Signals", icon="🔍", url_path="failure-signals"),
    st.Page(views.fail_risk, title="Fail Risk", icon="🎯", url_path="risk"),
    st.Page(views.equipment_health, title="Equipment Health", icon="🛠️", url_path="health"),
    st.Page(views.about, title="About", icon="ℹ️", url_path="about"),
]
page = st.navigation(pages)
views.sidebar_filters()
page.run()
