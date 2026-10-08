"""
visualization.py
----------------
Plotly figures used by the dashboard. Every function takes already-computed
results (no statistics are calculated here except plotting helpers such as the
fitted PMF curve), so the charts always reflect the real pipeline output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

PRIMARY = "#8B7CF6"
ACCENT = "#22D3EE"
MUTED = "#94A3B8"
GRID = "rgba(148,163,184,0.15)"
RISK_COLORS = {"LOW": "#34D399", "MEDIUM": "#FBBF24", "HIGH": "#F87171"}
DEMAND_COLORS = {"Low": "#60A5FA", "Medium": "#A78BFA", "High": "#F472B6"}


def _style(fig: go.Figure, title: str, x_title: str = "", y_title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
        height=height,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hoverlabel={"font": {"family": "Inter, sans-serif"}},
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


def arrival_distribution(observations, lam: float, title: str) -> go.Figure:
    """A. Histogram of observed counts with the fitted Poisson(lambda_hat) expected frequencies."""
    obs = np.asarray(observations, dtype=int)
    n = obs.size
    x = np.arange(max(0, obs.min() - 3), obs.max() + 4)
    observed = np.array([(obs == k).sum() for k in x])
    expected = n * stats.poisson.pmf(x, lam)
    fig = go.Figure()
    fig.add_bar(x=x, y=observed, name="Observed frequency", marker_color=PRIMARY, opacity=0.85,
                hovertemplate="x = %{x}<br>observed = %{y}<extra></extra>")
    fig.add_scatter(x=x, y=expected, name=f"Fitted Poisson (λ̂ = {lam:.2f}) × n", mode="lines+markers",
                    line={"color": ACCENT, "width": 3, "shape": "spline"}, marker={"size": 6},
                    hovertemplate="x = %{x}<br>expected = %{y:.2f}<extra></extra>")
    return _style(fig, title, "Arrivals in a 15-minute interval (x)", "Number of observations")


def lambda_by_period(table: pd.DataFrame, capacity: int) -> go.Figure:
    """B. lambda_hat per period with confidence interval, coloured by demand level."""
    fig = go.Figure()
    for level, color in DEMAND_COLORS.items():
        part = table[table["demand_level"] == level]
        if part.empty:
            continue
        fig.add_bar(x=part["period"], y=part["lambda_hat"], name=f"{level} demand", marker_color=color,
                    error_y={"type": "data", "symmetric": False, "color": "#CBD5E1", "thickness": 1.2,
                             "array": part["lambda_ci_high"] - part["lambda_hat"],
                             "arrayminus": part["lambda_hat"] - part["lambda_ci_low"]},
                    customdata=np.stack([part["lambda_ci_low"], part["lambda_ci_high"]], axis=-1),
                    hovertemplate="%{x}<br>λ̂ = %{y:.2f}<br>CI %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra></extra>")
    fig.add_hline(y=capacity, line_dash="dash", line_color=RISK_COLORS["HIGH"],
                  annotation_text=f"Capacity C = {capacity}", annotation_position="top left")
    fig.update_xaxes(categoryorder="array", categoryarray=table["period"].tolist(), tickangle=-45)
    return _style(fig, "Estimated arrival rate λ̂ by time period (error bars = confidence interval)",
                  "Time period", "λ̂ (mean arrivals per 15 min)")


def overload_by_period(table: pd.DataFrame, low: float, high: float) -> go.Figure:
    """C. P(X > C) per period coloured by risk level, with threshold lines."""
    fig = go.Figure()
    for risk, color in RISK_COLORS.items():
        part = table[table["risk"] == risk]
        if part.empty:
            continue
        fig.add_bar(x=part["period"], y=100 * part["overload_prob"], name=risk, marker_color=color,
                    hovertemplate="%{x}<br>P(X > C) = %{y:.2f}%<extra></extra>")
    fig.add_hline(y=100 * low, line_dash="dot", line_color=RISK_COLORS["MEDIUM"],
                  annotation_text=f"Low threshold {100 * low:.0f}%", annotation_position="top left")
    fig.add_hline(y=100 * high, line_dash="dot", line_color=RISK_COLORS["HIGH"],
                  annotation_text=f"High threshold {100 * high:.0f}%", annotation_position="top left")
    fig.update_xaxes(categoryorder="array", categoryarray=table["period"].tolist(), tickangle=-45)
    fig.update_yaxes(range=[0, 100])
    return _style(fig, "Probability of exceeding capacity, P(X > C)", "Time period", "P(X > C) in %")


def overload_curve(lam: float, capacity: int, period: str) -> go.Figure:
    """P(X > c) as a function of capacity c for one period (what-if view)."""
    c_values = np.arange(max(1, int(lam - 4 * np.sqrt(lam + 1))), int(lam + 5 * np.sqrt(lam + 1)) + 2)
    probs = 100 * stats.poisson.sf(c_values, lam)
    fig = go.Figure()
    fig.add_scatter(x=c_values, y=probs, mode="lines", fill="tozeroy", name="P(X > c)",
                    line={"color": PRIMARY, "width": 3, "shape": "spline"}, fillcolor="rgba(139,124,246,0.18)",
                    hovertemplate="capacity c = %{x}<br>P(X > c) = %{y:.2f}%<extra></extra>")
    fig.add_vline(x=capacity, line_dash="dash", line_color=ACCENT,
                  annotation_text=f"C = {capacity}", annotation_position="top right")
    fig.update_yaxes(range=[0, 100])
    return _style(fig, f"What-if: P(X > c) vs capacity — {period}", "Capacity c", "P(X > c) in %")


def risk_distribution(table: pd.DataFrame) -> go.Figure:
    """D. Number / share of periods in each risk level."""
    counts = table["risk"].value_counts().reindex(list(RISK_COLORS), fill_value=0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, sort=False,
                           marker={"colors": [RISK_COLORS[r] for r in counts.index]},
                           textinfo="label+value+percent",
                           hovertemplate="%{label}: %{value} periods (%{percent})<extra></extra>"))
    return _style(fig, "Risk distribution across periods", height=380)


def mean_vs_variance_scatter(table: pd.DataFrame) -> go.Figure:
    """E. Each period's mean vs variance; Poisson implies points near the line Var = Mean."""
    upper = float(np.nanmax([table["mean"].max(), table["variance"].max()]) * 1.1)
    fig = go.Figure()
    fig.add_scatter(x=[0, upper], y=[0, upper], mode="lines", name="Var = Mean (Poisson property)",
                    line={"color": MUTED, "dash": "dash"})
    colors = table["dispersion"].map({"Consistent with Poisson": RISK_COLORS["LOW"],
                                      "Overdispersed": RISK_COLORS["HIGH"],
                                      "Underdispersed": RISK_COLORS["MEDIUM"]}).fillna(MUTED)
    fig.add_scatter(x=table["mean"], y=table["variance"], mode="markers", name="Time period",
                    marker={"size": 11, "color": colors, "line": {"width": 1, "color": "#0F172A"}},
                    text=table["period"], customdata=np.stack([table["var_mean_ratio"], table["dispersion"]], axis=-1),
                    hovertemplate=("%{text}<br>mean = %{x:.2f}<br>variance = %{y:.2f}"
                                   "<br>Var/Mean = %{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>"))
    return _style(fig, "Mean vs variance (green = consistent with Poisson, red = overdispersed, amber = underdispersed)",
                  "Mean arrivals", "Sample variance")


def mean_vs_variance_bars(table: pd.DataFrame) -> go.Figure:
    """E (alternative view). Side-by-side mean and variance per period."""
    fig = go.Figure()
    fig.add_bar(x=table["period"], y=table["mean"], name="Mean", marker_color=PRIMARY)
    fig.add_bar(x=table["period"], y=table["variance"], name="Variance", marker_color=ACCENT)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-45)
    return _style(fig, "Mean and variance by time period", "Time period", "Value")


def arrivals_heatmap(df: pd.DataFrame) -> go.Figure:
    """F. Day x time-interval heatmap of arrival counts (gaps = missing records)."""
    pivot = df.pivot_table(index="date", columns="time_interval", values="arrivals", aggfunc="first").sort_index()
    labels = [f"{d:%a %d %b}" for d in pivot.index]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=labels, colorscale="Magma",
                               colorbar={"title": "Arrivals"},
                               hovertemplate="%{y}<br>%{x}<br>arrivals = %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    return _style(fig, "Arrivals heatmap: day × 15-minute interval", "Time interval", "Date",
                  height=max(420, 14 * len(labels) + 140))


def gof_observed_vs_expected(table: pd.DataFrame) -> go.Figure:
    """Observed vs expected frequencies for the (merged) chi-square categories."""
    fig = go.Figure()
    fig.add_bar(x=table["category"], y=table["observed"], name="Observed (O)", marker_color=PRIMARY)
    fig.add_bar(x=table["category"], y=table["expected"], name="Expected (E)", marker_color=ACCENT)
    fig.update_layout(barmode="group")
    return _style(fig, "Chi-square categories: observed vs expected frequencies", "Category (arrivals)", "Frequency",
                  height=380)
