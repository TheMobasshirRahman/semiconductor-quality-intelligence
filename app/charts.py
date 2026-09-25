"""
Plotly figure builders for the dashboard.

Colour roles follow a validated palette (colour-blind checked):
    identity  : Pass = blue, Fail = orange          (categorical slots 1-2)
    magnitude : one blue ramp, light -> dark        (sequential)
    polarity  : blue <-> red with a grey midpoint   (diverging)
    state     : critical red is reserved for alarms and always paired with a label
Rules: one y-axis per chart, thin marks, recessive grid, text never in series colour.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# categorical
PASS = "#2a78d6"
FAIL = "#eb6834"
THIRD = "#1baf7a"
# status (reserved)
CRITICAL = "#d03b3b"
WARNING = "#fab219"
GOOD = "#0ca30c"
# chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# ramps
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]            # Low, Medium, High
DIVERGING = [[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#c23b3b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
STATUS_COLORS = {"Pass": PASS, "Fail": FAIL}


def style(fig: go.Figure, height=360, title=None, legend=True) -> go.Figure:
    top = (40 if title else 10) + (34 if legend else 0)
    fig.update_layout(
        template="plotly_white", height=height,
        title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top", pad=dict(t=12, l=4)),
        font=dict(family=FONT, color=INK_2, size=13), title_font=dict(color=INK, size=15),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=top, b=10),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        bargap=0.15,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------- overview
def weekly_fail_rate(w: pd.DataFrame, overall: float, test_start=None) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=w["week"], y=w["fail_rate"] * 100, mode="lines+markers", name="Fail rate",
        line=dict(color=FAIL, width=2), marker=dict(size=8),
        customdata=np.c_[w["fails"], w["units"]],
        hovertemplate="Week of %{x|%d %b}<br>Fail rate %{y:.1f}%<br>%{customdata[0]} of %{customdata[1]} units<extra></extra>"))
    fig.add_hline(y=overall * 100, line=dict(color=MUTED, width=1),
                  annotation_text=f"overall {overall:.1%}", annotation_font_color=MUTED)
    _test_marker(fig, test_start)
    fig.update_yaxes(title="Fail rate %", rangemode="tozero")
    return style(fig, title="Weekly fail rate", legend=False)


def weekly_volume(w: pd.DataFrame, test_start=None) -> go.Figure:
    fig = go.Figure(go.Bar(x=w["week"], y=w["units"], marker=dict(color=PASS, cornerradius=4), name="Units",
                           hovertemplate="Week of %{x|%d %b}<br>%{y} units tested<extra></extra>"))
    _test_marker(fig, test_start)
    fig.update_yaxes(title="Units tested")
    return style(fig, height=260, title="Weekly volume", legend=False)


def _test_marker(fig, test_start):
    if test_start is not None:
        fig.add_vline(x=test_start, line=dict(color=MUTED, width=1))
        fig.add_annotation(x=test_start, y=1, yref="paper", text="test period →", showarrow=False,
                           xanchor="left", font=dict(color=MUTED, size=11))


# ---------------------------------------------------------------- SPC
def control_chart(units: pd.DataFrame, sensor: str, lim, flags: pd.DataFrame, test_start=None) -> go.Figure:
    fig = go.Figure()
    for status, color, symbol, size in [("Pass", PASS, "circle", 6), ("Fail", FAIL, "diamond", 9)]:
        d = units[units.status == status]
        fig.add_trace(go.Scatter(
            x=d["timestamp"], y=d[sensor], mode="markers", name=status,
            marker=dict(color=color, size=size, symbol=symbol, line=dict(color=SURFACE, width=1)),
            customdata=d[["unit_id"]],
            hovertemplate="%{customdata[0]}<br>%{x|%d %b %H:%M}<br>value %{y:.4g}<extra>" + status + "</extra>"))
    alarm = flags["alarm"].to_numpy()
    d = units[alarm]
    fig.add_trace(go.Scatter(
        x=d["timestamp"], y=d[sensor], mode="markers", name="⚠ Out-of-control alarm (R1/R2)",
        marker=dict(color="rgba(0,0,0,0)", size=15, line=dict(color=CRITICAL, width=2)),
        customdata=d[["unit_id"]], hovertemplate="ALARM %{customdata[0]}<br>value %{y:.4g}<extra></extra>"))
    for k, name, width in [(3, "UCL", 1.5), (-3, "LCL", 1.5), (0, "Center", 1)]:
        fig.add_hline(y=lim.level(k), line=dict(color=CRITICAL if k else MUTED, width=width),
                      annotation_text=f"{name} {lim.level(k):.4g}", annotation_position="top left",
                      annotation_font=dict(color=INK_2, size=11))
    for k in (2, -2):
        fig.add_hline(y=lim.level(k), line=dict(color=AXIS, width=1))
    lo, hi = lim.level(-3), lim.level(3)
    pad = (hi - lo) * 0.6
    vals = units[sensor].dropna()
    fig.update_yaxes(range=[max(vals.min(), lo - pad), min(vals.max(), hi + pad)], title=sensor)
    _test_marker(fig, test_start)
    return style(fig, height=430, title=f"Individuals control chart — {sensor}")


# ---------------------------------------------------------------- root cause
def importance_bar(ranking: pd.DataFrame, k=20) -> go.Figure:
    top = ranking.head(k).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["mean_rank"], y=top.index, orientation="h", marker=dict(color=PASS, cornerradius=4),
        customdata=np.c_[top["stability"] * 100, top["consensus_rank"]],
        text=[f"{s:.0%}" for s in top["stability"]], textposition="outside", textfont=dict(color=INK_2, size=11),
        hovertemplate="%{y}<br>rank #%{customdata[1]}<br>mean rank %{x:.1f}<br>stability %{customdata[0]:.0f}%<extra></extra>"))
    fig.update_xaxes(title="Mean rank across 5 methods (shorter = more important)")
    return style(fig, height=max(360, 22 * k), title=f"Top {k} sensors (label = stability)", legend=False)


def pass_fail_box(units: pd.DataFrame, sensor: str) -> go.Figure:
    fig = go.Figure()
    for status in ["Pass", "Fail"]:
        d = units[units.status == status]
        fig.add_trace(go.Box(x=d["period"] + " · " + d["timestamp"].dt.strftime("%b"), y=d[sensor], name=status,
                             marker=dict(color=STATUS_COLORS[status], size=4), line=dict(width=1.5),
                             boxpoints="outliers"))
    fig.update_layout(boxmode="group")
    q1, q99 = units[sensor].quantile([0.01, 0.99])
    fig.update_yaxes(range=[q1 - (q99 - q1) * 0.1, q99 + (q99 - q1) * 0.1], title=sensor)
    return style(fig, height=380, title=f"{sensor}: Pass vs Fail by month (1–99% range shown)")


def signature_heatmap(sig: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=sig.to_numpy(), x=sig.columns, y=sig.index, colorscale=DIVERGING, zmid=0, zmin=-1.5, zmax=1.5,
        text=np.round(sig.to_numpy(), 2), texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="σ", thickness=10),
        hovertemplate="%{y} · %{x}<br>Fail − Pass = %{z:.2f} σ<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, height=60 + 32 * len(sig), title="Fail signature by period (concept drift)", legend=False)


# ---------------------------------------------------------------- risk
def risk_band_bar(units: pd.DataFrame) -> go.Figure:
    order = ["Low", "Medium", "High"]
    g = units.groupby("risk_band")["fail"].agg(units="count", fails="sum", rate="mean").reindex(order).fillna(0)
    fig = go.Figure(go.Bar(
        x=order, y=g["rate"] * 100, marker=dict(color=ORDINAL_3, cornerradius=4),
        text=[f"{r:.1%}" for r in g["rate"]], textposition="outside", textfont=dict(color=INK_2),
        customdata=np.c_[g["fails"], g["units"]],
        hovertemplate="%{x} risk<br>fail rate %{y:.1f}%<br>%{customdata[0]:.0f} fails of %{customdata[1]:.0f} units<extra></extra>"))
    fig.update_yaxes(title="Actual fail rate %", rangemode="tozero")
    return style(fig, height=320, title="Actual fail rate by predicted risk band", legend=False)


def gain_chart(units: pd.DataFrame) -> go.Figure:
    d = units.sort_values("fail_prob", ascending=False)
    x = np.arange(1, len(d) + 1) / len(d) * 100
    y = d["fail"].cumsum().to_numpy() / max(d["fail"].sum(), 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random inspection",
                             line=dict(color=MUTED, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Model ranking", line=dict(color=PASS, width=2),
                             hovertemplate="inspect top %{x:.0f}% → catch %{y:.0f}% of fails<extra></extra>"))
    fig.update_xaxes(title="% of units inspected (highest risk first)", range=[0, 100])
    fig.update_yaxes(title="% of failures caught", range=[0, 102])
    return style(fig, height=360, title="Gain chart")


def contribution_bar(contrib: pd.Series, title: str, k=10, diverging=True) -> go.Figure:
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(k).iloc[::-1]
    colors = [FAIL if v > 0 else PASS for v in top] if diverging else [CRITICAL] * len(top)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="← lowers fail risk · raises fail risk →" if diverging else "share of anomaly statistic %")
    return style(fig, height=60 + 28 * len(top), title=title, legend=False)


# ---------------------------------------------------------------- health
def anomaly_timeline(units: pd.DataFrame, test_start=None) -> go.Figure:
    fig = go.Figure()
    for status, color, symbol, size in [("Pass", PASS, "circle", 6), ("Fail", FAIL, "diamond", 8)]:
        d = units[units.status == status]
        fig.add_trace(go.Scatter(
            x=d["timestamp"], y=d["anomaly_severity"], mode="markers", name=status,
            marker=dict(color=color, size=size, symbol=symbol, opacity=0.7, line=dict(color=SURFACE, width=1)),
            customdata=d[["unit_id", "top_sensors"]],
            hovertemplate="%{customdata[0]}<br>%{x|%d %b %H:%M}<br>%{y:,.2f} × limit<br>drivers: %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=CRITICAL, width=1.5), annotation_text="control limit",
                  annotation_font=dict(color=INK_2, size=11))
    fig.update_yaxes(type="log", title="max(T², SPE) ÷ limit")
    _test_marker(fig, test_start)
    return style(fig, height=400, title="Multivariate SPC — anomaly severity per unit")
