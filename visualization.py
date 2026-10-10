"""
visualization.py
----------------
Plotly figures used by the dashboard. Every function takes already-computed
results (no statistics are calculated here except plotting helpers such as the
fitted PMF curve), so the charts always reflect the real pipeline output.

All figures share one light "report" style (see `_style`) that matches the
Streamlit theme in .streamlit/config.toml. The app renders them with
`theme=None` so these colours are used exactly as written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# --- Palette taken from the project's review deck (navy, teal, blue, orange, green) ---
INK = "#1B2A4A"          # navy: titles and emphasis
GRAPHITE = "#5B6576"
RULE = "#DDE3EA"
GRID = "rgba(91,101,118,0.14)"
TEAL = "#00868B"
BLUE = TEAL              # primary series colour
BLUE_SOFT = "#9FD4D6"    # secondary / observed series
PRIMARY = TEAL
ACCENT = "#3366AA"
MUTED = "#8A94A3"
RISK_COLORS = {"LOW": "#2E8B57", "MEDIUM": "#E07B2E", "HIGH": "#C0392B"}
DEMAND_COLORS = {"Low": "#A8DADB", "Medium": "#3BA3A6", "High": "#1B2A4A"}
FONT = "IBM Plex Sans, system-ui, sans-serif"


def _style(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"} if title else None,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": INK, "size": 13},
        height=height,
        margin={"l": 56, "r": 16, "t": 56 if title else 24, "b": 56},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0,
                "font": {"color": GRAPHITE}},
        hoverlabel={"font": {"family": FONT}, "bgcolor": "#FFFFFF", "bordercolor": RULE},
        bargap=0.25,
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, linecolor=RULE, tickfont={"color": GRAPHITE},
                     title_font={"color": GRAPHITE, "size": 12})
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, linecolor=RULE, tickfont={"color": GRAPHITE},
                     title_font={"color": GRAPHITE, "size": 12})
    return fig


def _period_axis(fig: go.Figure, periods: list[str]) -> None:
    fig.update_xaxes(categoryorder="array", categoryarray=periods, tickangle=-35 if len(periods) > 6 else 0)


def arrival_distribution(observations, lam: float, title: str) -> go.Figure:
    """Histogram of observed counts with the fitted Poisson(lambda_hat) expected frequencies."""
    obs = np.asarray(observations, dtype=int)
    n = obs.size
    x = np.arange(max(0, obs.min() - 3), obs.max() + 4)
    observed = np.array([(obs == k).sum() for k in x])
    expected = n * stats.poisson.pmf(x, lam)
    fig = go.Figure()
    fig.add_bar(x=x, y=observed, name="Days observed with x arrivals", marker_color=BLUE_SOFT,
                hovertemplate="x = %{x} arrivals<br>seen on %{y} days<extra></extra>")
    fig.add_scatter(x=x, y=expected, name=f"Poisson model, λ̂ = {lam:.2f}", mode="lines+markers",
                    line={"color": BLUE, "width": 2.5, "shape": "spline"}, marker={"size": 5},
                    hovertemplate="x = %{x}<br>model expects %{y:.2f} days<extra></extra>")
    return _style(fig, title, "Arrivals in the slot (x)", "Number of days")


def lambda_by_period(table: pd.DataFrame, capacity: int, selected: str | None = None) -> go.Figure:
    """lambda_hat per period with confidence interval, coloured by demand level; selected bar outlined."""
    fig = go.Figure()
    for level, color in DEMAND_COLORS.items():
        part = table[table["demand_level"] == level]
        if part.empty:
            continue
        outline = [3 if p == selected else 0 for p in part["period"]]
        fig.add_bar(x=part["period"], y=part["lambda_hat"], name=f"{level} demand", marker_color=color,
                    marker_line={"color": INK, "width": outline},
                    error_y={"type": "data", "symmetric": False, "color": GRAPHITE, "thickness": 1.2, "width": 5,
                             "array": part["lambda_ci_high"] - part["lambda_hat"],
                             "arrayminus": part["lambda_hat"] - part["lambda_ci_low"]},
                    customdata=np.stack([part["lambda_ci_low"], part["lambda_ci_high"]], axis=-1),
                    hovertemplate="%{x}<br>λ̂ = %{y:.2f}<br>CI %{customdata[0]:.2f} to %{customdata[1]:.2f}"
                                  "<br><i>click to inspect</i><extra></extra>")
    fig.add_hline(y=capacity, line_dash="dash", line_color=RISK_COLORS["HIGH"], line_width=1.5,
                  annotation_text=f"capacity C = {capacity}", annotation_position="top left",
                  annotation_font_color=RISK_COLORS["HIGH"])
    _period_axis(fig, table["period"].tolist())
    fig.update_layout(clickmode="event+select", dragmode=False)
    return _style(fig, "", "Time slot", "λ̂, mean arrivals per 15 min", height=380)


def overload_by_period(table: pd.DataFrame, low: float, high: float) -> go.Figure:
    """P(X > C) per period coloured by risk level, with threshold lines."""
    fig = go.Figure()
    for risk, color in RISK_COLORS.items():
        part = table[table["risk"] == risk]
        if part.empty:
            continue
        fig.add_bar(x=part["period"], y=100 * part["overload_prob"], name=risk.capitalize() + " risk",
                    marker_color=color,
                    customdata=np.stack([100 * part["overload_prob_low"], 100 * part["overload_prob_high"]], axis=-1),
                    error_y={"type": "data", "symmetric": False, "color": GRAPHITE, "thickness": 1, "width": 4,
                             "array": 100 * (part["overload_prob_high"] - part["overload_prob"]),
                             "arrayminus": 100 * (part["overload_prob"] - part["overload_prob_low"])},
                    hovertemplate="%{x}<br>P(X > C) = %{y:.2f}%<br>range over λ CI: "
                                  "%{customdata[0]:.1f}% to %{customdata[1]:.1f}%<extra></extra>")
    for value, label, color in ((low, "low threshold", RISK_COLORS["MEDIUM"]),
                                (high, "high threshold", RISK_COLORS["HIGH"])):
        fig.add_hline(y=100 * value, line_dash="dot", line_color=color, line_width=1.5,
                      annotation_text=f"{label} {100 * value:.0f}%", annotation_position="top right",
                      annotation_font_color=color)
    _period_axis(fig, table["period"].tolist())
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_layout(dragmode=False)
    return _style(fig, "", "Time slot", "P(X > C)", height=380)


def overload_curve(lam: float, capacity: int, period: str) -> go.Figure:
    """P(X > c) as a function of capacity c for one period (what-if view)."""
    c_values = np.arange(max(1, int(lam - 4 * np.sqrt(lam + 1))), int(lam + 5 * np.sqrt(lam + 1)) + 2)
    probs = 100 * stats.poisson.sf(c_values, lam)
    fig = go.Figure()
    fig.add_scatter(x=c_values, y=probs, mode="lines", fill="tozeroy", name="P(X > c)",
                    line={"color": BLUE, "width": 2.5, "shape": "hv"}, fillcolor="rgba(0,134,139,0.10)",
                    hovertemplate="capacity c = %{x}<br>P(X > c) = %{y:.2f}%<extra></extra>")
    fig.add_vline(x=capacity, line_dash="dash", line_color=RISK_COLORS["HIGH"], line_width=1.5,
                  annotation_text=f"C = {capacity}", annotation_position="top right",
                  annotation_font_color=RISK_COLORS["HIGH"])
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return _style(fig, f"If capacity changed: {period}", "Capacity c", "P(X > c)", height=380)


def risk_distribution(table: pd.DataFrame) -> go.Figure:
    """Number / share of periods in each risk level."""
    counts = table["risk"].value_counts().reindex(list(RISK_COLORS), fill_value=0)
    fig = go.Figure(go.Pie(labels=[r.capitalize() for r in counts.index], values=counts.values, hole=0.6, sort=False,
                           marker={"colors": [RISK_COLORS[r] for r in counts.index], "line": {"color": "#FFFFFF", "width": 2}},
                           textinfo="label+value",
                           hovertemplate="%{label} risk: %{value} slots (%{percent})<extra></extra>"))
    return _style(fig, "Slots in each risk level", height=340)


def mean_vs_variance_scatter(table: pd.DataFrame) -> go.Figure:
    """Each period's mean vs variance; Poisson implies points near the line Var = Mean."""
    upper = float(np.nanmax([table["mean"].max(), table["variance"].max()]) * 1.12)
    fig = go.Figure()
    fig.add_scatter(x=[0, upper], y=[0, upper], mode="lines", name="Var = Mean (Poisson property)",
                    line={"color": MUTED, "dash": "dash", "width": 1.5}, hoverinfo="skip")
    colors = table["dispersion"].map({"Consistent with Poisson": RISK_COLORS["LOW"],
                                      "Overdispersed": RISK_COLORS["HIGH"],
                                      "Underdispersed": RISK_COLORS["MEDIUM"]}).fillna(MUTED)
    fig.add_scatter(x=table["mean"], y=table["variance"], mode="markers+text", name="Time slot",
                    marker={"size": 12, "color": colors, "line": {"width": 1.5, "color": "#FFFFFF"}},
                    text=table["period"].str.slice(0, 5), textposition="top center",
                    textfont={"size": 10, "color": GRAPHITE},
                    customdata=np.stack([table["period"], table["var_mean_ratio"], table["dispersion"]], axis=-1),
                    hovertemplate=("%{customdata[0]}<br>mean = %{x:.2f}<br>variance = %{y:.2f}"
                                   "<br>Var/Mean = %{customdata[1]:.2f}<br>%{customdata[2]}<extra></extra>"))
    fig.update_xaxes(range=[0, upper])
    fig.update_yaxes(range=[0, upper])
    return _style(fig, "", "Sample mean", "Sample variance", height=400)


def mean_vs_variance_bars(table: pd.DataFrame) -> go.Figure:
    """Side-by-side mean and variance per period."""
    fig = go.Figure()
    fig.add_bar(x=table["period"], y=table["mean"], name="Mean", marker_color=BLUE)
    fig.add_bar(x=table["period"], y=table["variance"], name="Variance", marker_color=BLUE_SOFT)
    fig.update_layout(barmode="group")
    _period_axis(fig, table["period"].tolist())
    return _style(fig, "", "Time slot", "Value", height=400)


def arrivals_heatmap(df: pd.DataFrame) -> go.Figure:
    """Day x time-interval heatmap of arrival counts (gaps = missing records)."""
    pivot = df.pivot_table(index="date", columns="time_interval", values="arrivals", aggfunc="first").sort_index()
    labels = [f"{d:%a %d %b}" for d in pivot.index]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=labels,
                               colorscale=[[0, "#EAF6F6"], [0.5, "#3BA3A6"], [1, "#1B2A4A"]],
                               colorbar={"thickness": 12, "ticksuffix": " "},
                               hovertemplate="%{y}<br>%{x}<br>arrivals = %{z}<extra></extra>", xgap=1, ygap=1))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _style(fig, "", "Time slot", "", height=max(420, 7 * len(labels) + 120))


def gof_observed_vs_expected(table: pd.DataFrame) -> go.Figure:
    """Observed vs expected frequencies for the (merged) chi-square categories."""
    fig = go.Figure()
    fig.add_bar(x=table["category"], y=table["observed"], name="Observed O", marker_color=BLUE)
    fig.add_bar(x=table["category"], y=table["expected"], name="Expected E", marker_color=BLUE_SOFT)
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category")
    return _style(fig, "", "Category (arrivals)", "Days", height=360)


# ---------------------------------------------------------------------------
# Interactive and dynamic views
# ---------------------------------------------------------------------------
def poisson_explorer(lam: float, capacity: int) -> go.Figure:
    """PMF of Poisson(lambda) with the overload tail X > C shaded."""
    hi = int(max(capacity + 3, lam + 5 * np.sqrt(lam + 1) + 3))
    x = np.arange(0, hi + 1)
    pmf = stats.poisson.pmf(x, lam)
    colors = np.where(x > capacity, RISK_COLORS["HIGH"], BLUE_SOFT)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color=colors, name="P(X = x)",
                           hovertemplate="P(X = %{x}) = %{y:.4f}<extra></extra>"))
    fig.add_vline(x=capacity + 0.5, line_color=INK, line_width=1.5, line_dash="dash",
                  annotation_text=f"C = {capacity}", annotation_position="top right")
    fig.update_layout(showlegend=False, bargap=0.08)
    return _style(fig, "", "Arrivals in one slot (x)", "P(X = x)", height=340)


def monte_carlo_convergence(running: np.ndarray, p_model: float, frames: int = 40) -> go.Figure:
    """Animated running share of simulated days that overflowed, converging to P(X > C)."""
    n = running.size
    days = np.arange(1, n + 1)
    cut_points = np.unique(np.geomspace(1, n, num=min(frames, n)).astype(int))
    y_max = max(100 * float(running.max()), 100 * p_model) * 1.15 + 1

    def trace(k: int) -> go.Scatter:
        return go.Scatter(x=days[:k], y=100 * running[:k], mode="lines", name="Simulated share of days",
                          line={"color": BLUE, "width": 2}, hovertemplate="after %{x} days: %{y:.2f}%<extra></extra>")

    fig = go.Figure(data=[trace(n)],  # full path shown until Play replays it from day 1
                    frames=[go.Frame(data=[trace(int(k))], name=str(k)) for k in cut_points])
    fig.add_hline(y=100 * p_model, line_dash="dash", line_color=RISK_COLORS["HIGH"], line_width=1.5,
                  annotation_text=f"model P(X > C) = {100 * p_model:.2f}%", annotation_position="top right",
                  annotation_font_color=RISK_COLORS["HIGH"])
    fig.update_xaxes(range=[0, n * 1.02])
    fig.update_yaxes(range=[0, y_max], ticksuffix="%")
    fig.update_layout(updatemenus=[{
        "type": "buttons", "showactive": False, "x": 1, "xanchor": "right", "y": -0.22, "yanchor": "top",
        "direction": "left", "pad": {"t": 0, "r": 0},
        "buttons": [
            {"label": "Play", "method": "animate",
             "args": [None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": False,
                             "transition": {"duration": 0}}]},
            {"label": "Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
        ]}])
    fig = _style(fig, "", "Simulated lunch services (days)", "Share of days with X > C", height=380)
    fig.update_layout(margin={"b": 90})
    return fig


def simulated_day(periods: list[str], counts: np.ndarray, lambdas: np.ndarray, capacity: int) -> go.Figure:
    """One simulated lunch service: a Poisson draw per slot against capacity."""
    over = counts > capacity
    fig = go.Figure()
    fig.add_bar(x=periods, y=counts, name="Simulated arrivals",
                marker_color=np.where(over, RISK_COLORS["HIGH"], BLUE_SOFT),
                customdata=lambdas, hovertemplate="%{x}<br>%{y} arrivals (λ̂ = %{customdata:.1f})<extra></extra>")
    fig.add_scatter(x=periods, y=lambdas, mode="markers", name="Expected λ̂",
                    marker={"symbol": "line-ew-open", "size": 26, "color": INK, "line": {"width": 2}},
                    hoverinfo="skip")
    fig.add_hline(y=capacity, line_dash="dash", line_color=RISK_COLORS["HIGH"], line_width=1.5,
                  annotation_text=f"capacity C = {capacity}", annotation_position="top left",
                  annotation_font_color=RISK_COLORS["HIGH"])
    _period_axis(fig, periods)
    return _style(fig, "", "Time slot", "Arrivals", height=360)


def quiet_ranking(ranking: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    """Slots ordered from quietest to busiest with confidence intervals; the answer highlighted."""
    ranked = ranking.iloc[::-1]  # quietest at the top of a horizontal bar chart
    colors = [RISK_COLORS["LOW"] if p == highlight else BLUE_SOFT for p in ranked["period"]]
    fig = go.Figure(go.Bar(
        y=ranked["period"], x=ranked["lambda_hat"], orientation="h", marker_color=colors,
        error_x={"type": "data", "symmetric": False, "color": GRAPHITE, "thickness": 1.2, "width": 4,
                 "array": ranked["lambda_ci_high"] - ranked["lambda_hat"],
                 "arrayminus": ranked["lambda_hat"] - ranked["lambda_ci_low"]},
        customdata=100 * ranked["overload_prob"],
        hovertemplate="%{y}<br>λ̂ = %{x:.2f}<br>P(X > C) = %{customdata:.2f}%<extra></extra>"))
    fig.update_yaxes(type="category", showgrid=False)
    fig.update_layout(showlegend=False)
    return _style(fig, "", "λ̂, expected arrivals per 15 min", "", height=60 + 34 * len(ranked))


def comfort_bars(probabilities: pd.Series, max_arrivals: int, highlight: str | None = None) -> go.Figure:
    """P(X <= k) per slot: the chance a visitor meets at most k fellow arrivals in that slot."""
    colors = [RISK_COLORS["LOW"] if p == highlight else BLUE for p in probabilities.index]
    fig = go.Figure(go.Bar(x=probabilities.index, y=100 * probabilities.values, marker_color=colors,
                           hovertemplate="%{x}<br>P(X ≤ " + str(max_arrivals) + ") = %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    _period_axis(fig, list(probabilities.index))
    fig.update_layout(showlegend=False)
    return _style(fig, "", "Time slot", f"P(at most {max_arrivals} arrivals)", height=340)


def weekday_heatmap(rates: pd.DataFrame) -> go.Figure:
    """Mean arrivals for every weekday x slot, with each weekday's quietest slot marked."""
    z = rates.values
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.1f}")(np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(z=z, x=list(rates.columns), y=list(rates.index), text=text, texttemplate="%{text}",
                               colorscale=[[0, "#DFF1E7"], [0.5, "#FBE3CF"], [1, "#F1B7B0"]], showscale=False,
                               xgap=2, ygap=2, hovertemplate="%{y}, %{x}<br>mean arrivals %{z:.2f}<extra></extra>"))
    best_cols = np.nanargmin(np.where(np.isnan(z), np.inf, z), axis=1)
    fig.add_scatter(x=[rates.columns[c] for c in best_cols], y=list(rates.index), mode="markers",
                    marker={"symbol": "square-open", "size": 34, "color": INK, "line": {"width": 2}},
                    name="Quietest slot that weekday", hoverinfo="skip")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _style(fig, "", "Time slot", "", height=110 + 46 * len(rates))


# ---------------------------------------------------------------------------
# Exploratory data analysis and planning views
# ---------------------------------------------------------------------------
def slot_boxplot(df: pd.DataFrame, group_col: str = "time_interval") -> go.Figure:
    """Distribution of arrival counts in every slot (box = middle 50%, line = median, dot = mean)."""
    periods = sorted(df[group_col].unique())
    fig = go.Figure()
    for period in periods:
        values = df.loc[df[group_col] == period, "arrivals"]
        fig.add_box(y=values, name=period, marker_color=TEAL, line_color=INK, fillcolor="rgba(0,134,139,0.18)",
                    boxmean=True, boxpoints="outliers", showlegend=False,
                    hovertemplate=period + "<br>%{y} arrivals<extra></extra>")
    _period_axis(fig, periods)
    return _style(fig, "", "Time slot", "Arrivals per slot", height=380)


def weekday_profile(df: pd.DataFrame, group_col: str = "time_interval") -> go.Figure:
    """Mean arrivals per slot, one line per weekday (click legend entries to hide or isolate days)."""
    pivot = df.pivot_table(index="day", columns=group_col, values="arrivals", aggfunc="mean")
    order = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
             if d in pivot.index]
    palette = [INK, TEAL, ACCENT, RISK_COLORS["MEDIUM"], RISK_COLORS["LOW"], MUTED, RISK_COLORS["HIGH"]]
    fig = go.Figure()
    for day, color in zip(order, palette):
        fig.add_scatter(x=list(pivot.columns), y=pivot.loc[day].values, mode="lines+markers", name=day,
                        line={"color": color, "width": 2}, marker={"size": 6},
                        hovertemplate=day + "<br>%{x}<br>mean %{y:.1f} arrivals<extra></extra>")
    _period_axis(fig, list(pivot.columns))
    return _style(fig, "", "Time slot", "Mean arrivals", height=380)


def required_capacity_chart(periods: list[str], needed: list[int], capacity: int, target: float) -> go.Figure:
    """Capacity each slot needs to keep P(overload) below the target, against the current capacity."""
    colors = [RISK_COLORS["HIGH"] if n > capacity else RISK_COLORS["LOW"] for n in needed]
    fig = go.Figure(go.Bar(x=periods, y=needed, marker_color=colors, text=needed, textposition="outside",
                           hovertemplate="%{x}<br>needs capacity %{y}<extra></extra>"))
    fig.add_hline(y=capacity, line_dash="dash", line_color=INK, line_width=1.5,
                  annotation_text=f"current capacity {capacity}", annotation_position="top right")
    _period_axis(fig, periods)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, max(max(needed), capacity) * 1.18])
    return _style(fig, "", "Time slot", f"Capacity for P(overload) < {100 * target:.0f}%", height=360)
