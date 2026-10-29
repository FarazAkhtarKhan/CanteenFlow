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


# ===========================================================================
# 3D views and forecast charts (research gaps G7-G11)
# ===========================================================================
# One hue, light -> dark, for magnitudes (arrivals, probabilities) on every 3D surface.
SEQUENTIAL = [[0.0, "#E3F4F4"], [0.35, "#7CC4C6"], [0.7, "#00868B"], [1.0, INK]]
CAMERA = {"eye": {"x": 1.55, "y": -1.55, "z": 0.95}}


def _axis3d(title: str, extra: dict | None = None) -> dict:
    return {"title": {"text": title, "font": {"color": GRAPHITE, "size": 12}}, "gridcolor": "rgba(91,101,118,0.25)",
            "backgroundcolor": "rgba(242,245,248,0.6)", "showbackground": True, "zeroline": False,
            "tickfont": {"color": GRAPHITE, "size": 10}} | (extra or {})


def _style3d(fig: go.Figure, x_title: str, y_title: str, z_title: str, height: int = 520,
             x: dict | None = None, y: dict | None = None, z: dict | None = None) -> go.Figure:
    """Shared 3D scene style; x / y / z hold extra axis settings (ticks, ranges)."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT, "color": INK, "size": 12},
        height=height, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        hoverlabel={"font": {"family": FONT}, "bgcolor": "#FFFFFF", "bordercolor": RULE},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0,
                "font": {"color": GRAPHITE}},
        scene={"xaxis": _axis3d(x_title, x), "yaxis": _axis3d(y_title, y),
               "zaxis": _axis3d(z_title, z), "camera": CAMERA, "aspectmode": "manual",
               "aspectratio": {"x": 1.25, "y": 1.0, "z": 0.65}})
    return fig


def overload_surface_3d(table: pd.DataFrame, capacity: int) -> go.Figure:
    """P(X > C) as a surface over arrival rate λ and capacity C, with every slot placed on it."""
    lam_hi = float(max(table["lambda_hat"].max() * 1.35, capacity * 1.2))
    lam = np.linspace(max(1.0, table["lambda_hat"].min() * 0.6), lam_hi, 60)
    caps = np.arange(max(1, int(capacity * 0.5)), int(capacity * 1.6) + 1)
    L, C = np.meshgrid(lam, caps)
    P = 100 * stats.poisson.sf(C, L)
    fig = go.Figure(go.Surface(
        x=L, y=C, z=P, colorscale=SEQUENTIAL, opacity=0.92, showscale=False,
        hovertemplate="λ = %{x:.1f}, C = %{y}<br>P(X > C) = %{z:.1f}%<extra></extra>",
        contours={"z": {"show": True, "start": 0, "end": 100, "size": 10, "color": "rgba(255,255,255,0.7)"}}))
    fig.add_scatter3d(
        x=table["lambda_hat"], y=[capacity] * len(table), z=100 * table["overload_prob"], mode="markers+text",
        text=[p[:5] for p in table["period"]], textposition="top center", textfont={"size": 10, "color": INK},
        marker={"size": 6, "color": [RISK_COLORS[r] for r in table["risk"]], "line": {"color": "#FFFFFF", "width": 2}},
        name=f"Slots at C = {capacity}", customdata=table["period"],
        hovertemplate="%{customdata}<br>λ̂ = %{x:.2f}<br>P(X > %{y}) = %{z:.1f}%<extra></extra>")
    return _style3d(fig, "Arrival rate λ", "Capacity C", "P(X > C), %", z={"range": [0, 100]})


def forecast_surface_3d(probabilities: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    """Predicted P(X > C) by weekday (y) and slot (x) for one day type."""
    periods, days = list(probabilities.columns), list(probabilities.index)
    z = 100 * probabilities.to_numpy(float)
    fig = go.Figure(go.Surface(
        z=z, x=list(range(len(periods))), y=list(range(len(days))), colorscale=SEQUENTIAL, cmin=0, cmax=100,
        colorbar={"title": {"text": "P(X > C), %", "side": "right"}, "thickness": 12, "len": 0.7},
        customdata=np.dstack([np.tile(periods, (len(days), 1)), np.tile(np.array(days)[:, None], (1, len(periods)))]),
        hovertemplate="%{customdata[1]}, %{customdata[0]}<br>P(X > C) = %{z:.1f}%<extra></extra>"))
    if highlight in days:
        i = days.index(highlight)
        fig.add_scatter3d(x=list(range(len(periods))), y=[i] * len(periods), z=z[i] + 1.5, mode="lines+markers",
                          line={"color": RISK_COLORS["HIGH"], "width": 6}, marker={"size": 3, "color": RISK_COLORS["HIGH"]},
                          name=f"Selected day: {highlight}", hoverinfo="skip")
    return _style3d(fig, "Time slot", "Weekday", "P(X > C), %",
                    x={"tickvals": list(range(len(periods))), "ticktext": [p[:5] for p in periods]},
                    y={"tickvals": list(range(len(days))), "ticktext": [d[:3] for d in days]},
                    z={"range": [0, 100]})


def update_surface_3d(mu: pd.Series, capacity: int, var_w: float, observed_slots: int,
                      ratios=np.linspace(0.7, 1.4, 29)) -> go.Figure:
    """How the overload risk of the remaining slots moves with how busy the day has been so far (G9)."""
    from forecast import update_day_factor, updated_overload  # local import keeps visualization independent
    seen, rest = mu.iloc[:observed_slots], mu.iloc[observed_slots:]
    z = []
    for r in ratios:
        shape, rate = update_day_factor(seen.to_numpy(), r * seen.to_numpy(), var_w)
        z.append(100 * updated_overload(rest.to_numpy(), capacity, shape, rate))
    z = np.array(z)
    fig = go.Figure(go.Surface(
        z=z, x=list(range(len(rest))), y=100 * np.asarray(ratios), colorscale=SEQUENTIAL, cmin=0, cmax=100,
        colorbar={"title": {"text": "Updated P(X > C), %", "side": "right"}, "thickness": 12, "len": 0.7},
        customdata=np.tile(np.array(rest.index), (len(ratios), 1)),
        hovertemplate="%{customdata}<br>so far %{y:.0f}% of forecast<br>P(X > C) = %{z:.1f}%<extra></extra>"))
    base_row = int(np.argmin(np.abs(np.asarray(ratios) - 1.0)))
    fig.add_scatter3d(x=list(range(len(rest))), y=[100.0] * len(rest), z=z[base_row] + 1, mode="lines",
                      line={"color": INK, "width": 6, "dash": "dash"}, name="Day exactly as forecast", hoverinfo="skip")
    return _style3d(fig, "Remaining slot", "Arrivals so far, % of forecast", "P(X > C), %",
                    x={"tickvals": list(range(len(rest))), "ticktext": [p[:5] for p in rest.index]},
                    z={"range": [0, 100]})


def canteen_scene_3d(periods: list[str], counts: np.ndarray, mu: np.ndarray, capacity: int,
                     probabilities: np.ndarray, seed: int = 0) -> go.Figure:
    """Animated 3D canteen: one marker per simulated student in each slot.

    The first C students of a slot queue at the two serving counters (teal); anyone beyond capacity
    waits as overflow near the entrance (red). Play steps through the lunch period slot by slot.
    """
    rng = np.random.default_rng(seed)
    width, depth = 20.0, 14.0

    def box(x0, x1, y0, y1, z1, color, name):
        xs = [x0, x1, x1, x0, x0, x1, x1, x0]
        ys = [y0, y0, y1, y1, y0, y0, y1, y1]
        zs = [0, 0, 0, 0, z1, z1, z1, z1]
        return go.Mesh3d(x=xs, y=ys, z=zs, i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3], j=[1, 2, 5, 6, 1, 4, 2, 5, 3, 6, 0, 7],
                         k=[2, 3, 6, 7, 4, 5, 5, 6, 6, 7, 7, 4], color=color, opacity=0.85, name=name,
                         hoverinfo="name", showlegend=False, flatshading=True)

    static = [box(0, width, 0, depth, 0.02, "#F2F5F8", "Canteen floor"),
              box(3, 8, depth - 1.6, depth - 0.6, 1.1, INK, "Serving counter 1"),
              box(12, 17, depth - 1.6, depth - 0.6, 1.1, INK, "Serving counter 2"),
              box(8.5, 11.5, -0.25, 0.05, 0.25, MUTED, "Entrance")]
    for tx in (4.5, 9.5, 14.5):  # dining tables
        for ty in (3.0, 6.0):
            static.append(box(tx - 1.0, tx + 1.0, ty - 0.5, ty + 0.5, 0.75, "#C9D3DE", "Dining table"))

    def students(n: int):
        served = min(n, capacity)
        lanes = np.array([5.5, 14.5])
        k = np.arange(served)
        x = lanes[k % 2] + rng.normal(0, 0.18, served)
        y = depth - 2.2 - 0.55 * (k // 2) + rng.normal(0, 0.08, served)
        extra = n - served
        ox = 10 + rng.normal(0, 1.4, extra)
        oy = 1.2 + np.abs(rng.normal(0, 0.9, extra))
        xs, ys = np.concatenate([x, ox]), np.concatenate([y, oy])
        colors = [TEAL] * served + [RISK_COLORS["HIGH"]] * extra
        return go.Scatter3d(x=xs, y=ys, z=np.full(n, 1.65), mode="markers", name="Students",
                            marker={"size": 6, "color": colors, "line": {"color": "#FFFFFF", "width": 1}},
                            hovertemplate="student<extra></extra>", showlegend=False)

    frames = []
    for p, n, m, prob in zip(periods, counts, mu, probabilities):
        title = (f"{p.replace('-', '–')} · {int(n)} students arrive (about {m:.0f} expected) · "
                 f"chance of overcrowding {100 * prob:.0f}%"
                 + (f" · {int(n) - capacity} over capacity" if n > capacity else ""))
        frames.append(go.Frame(data=[students(int(n))], name=p, traces=[len(static)],
                               layout={"title": {"text": title}}))
    fig = go.Figure(data=static + [frames[0].data[0]], frames=frames)
    fig.update_layout(
        title={"text": frames[0].layout.title.text, "font": {"size": 14, "color": INK}, "x": 0, "xanchor": "left"},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "xanchor": "left", "y": 0, "yanchor": "top",
                      "direction": "left", "buttons": [
                          {"label": "Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 1100, "redraw": True}, "fromcurrent": True,
                                           "transition": {"duration": 0}}]},
                          {"label": "Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}]}],
        sliders=[{"active": 0, "x": 0.24, "len": 0.76, "y": 0, "yanchor": "top", "pad": {"t": 0},
                  "currentvalue": {"prefix": "Slot: ", "font": {"color": INK}},
                  "steps": [{"label": p[:5], "method": "animate",
                             "args": [[p], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                            for p in periods]}])
    fig = _style3d(fig, "", "", "", height=480,
                   x={"range": [-0.5, width + 0.5], "showticklabels": False, "title": {"text": ""}},
                   y={"range": [-1, depth + 0.5], "showticklabels": False, "title": {"text": ""}},
                   z={"range": [0, 4], "showticklabels": False, "title": {"text": ""}})
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 60},
                      scene={"aspectratio": {"x": 1.7, "y": 1.2, "z": 0.35},
                             "camera": {"eye": {"x": 0.25, "y": -1.15, "z": 0.85}}})
    return fig


def peak_probability_bars(p_peak: pd.Series) -> go.Figure:
    """P(slot t has the most arrivals of the day), from the simulation."""
    best = p_peak.idxmax()
    colors = [INK if p == best else BLUE_SOFT for p in p_peak.index]
    fig = go.Figure(go.Bar(x=list(p_peak.index), y=100 * p_peak.to_numpy(), marker_color=colors,
                           text=[f"{100 * v:.0f}%" if v >= 0.05 else "" for v in p_peak], textposition="outside",
                           textfont={"color": INK}, hovertemplate="%{x}<br>P(peak) = %{y:.1f}%<extra></extra>"))
    _period_axis(fig, list(p_peak.index))
    fig.update_yaxes(ticksuffix="%", range=[0, max(100 * p_peak.max() * 1.2, 10)])
    fig.update_layout(showlegend=False)
    return _style(fig, "", "Time slot", "P(slot is the day's peak)", height=340)


def update_before_after(table: pd.DataFrame) -> go.Figure:
    """Overload probability of the remaining slots before and after today's counts (G9)."""
    fig = go.Figure()
    fig.add_bar(x=table["period"], y=100 * table["p_before"], name="Forecast at start of day", marker_color=BLUE_SOFT,
                hovertemplate="%{x}<br>before: %{y:.1f}%<extra></extra>")
    fig.add_bar(x=table["period"], y=100 * table["p_after"], name="Updated with today's counts", marker_color=INK,
                hovertemplate="%{x}<br>after: %{y:.1f}%<extra></extra>")
    _period_axis(fig, list(table["period"]))
    fig.update_yaxes(ticksuffix="%", range=[0, 100])
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    return _style(fig, "", "Remaining slot", "P(X > C)", height=340)


def coefficient_chart(coef: pd.DataFrame) -> go.Figure:
    """Regression effects as multipliers on expected arrivals, with 95% intervals (reference level = 1)."""
    part = coef[coef["term"] != "Intercept"].iloc[::-1]
    lo, hi = np.exp(part["estimate"] - 1.96 * part["std_error"]), np.exp(part["estimate"] + 1.96 * part["std_error"])
    colors = [GRAPHITE if p >= 0.05 else (TEAL if m > 1 else ACCENT) for p, m in zip(part["p_value"], part["multiplier"])]
    fig = go.Figure(go.Scatter(
        x=part["multiplier"], y=part["term"], mode="markers", marker={"size": 10, "color": colors},
        error_x={"type": "data", "symmetric": False, "array": hi - part["multiplier"],
                 "arrayminus": part["multiplier"] - lo, "color": GRAPHITE, "thickness": 1.2, "width": 0},
        customdata=np.stack([part["t"], part["p_value"]], axis=-1),
        hovertemplate="%{y}<br>× %{x:.3f}<br>t = %{customdata[0]:.2f}, p = %{customdata[1]:.3g}<extra></extra>"))
    fig.add_vline(x=1, line_color=INK, line_width=1, line_dash="dot")
    fig.update_layout(showlegend=False)
    return _style(fig, "", "Multiplier on expected arrivals (1 = no effect)", "", height=max(320, 26 * len(part) + 80))


def poisson_vs_mixed_bars(periods: list[str], p_poisson, p_mixed) -> go.Figure:
    """Overload probability under the plain Poisson and the Poisson-Gamma model (G10)."""
    fig = go.Figure()
    fig.add_bar(x=periods, y=100 * np.asarray(p_poisson), name="Poisson", marker_color=BLUE_SOFT,
                hovertemplate="%{x}<br>Poisson: %{y:.2f}%<extra></extra>")
    fig.add_bar(x=periods, y=100 * np.asarray(p_mixed), name="Poisson–Gamma (day-to-day variation)",
                marker_color=INK, hovertemplate="%{x}<br>Poisson–Gamma: %{y:.2f}%<extra></extra>")
    _period_axis(fig, periods)
    fig.update_yaxes(ticksuffix="%", range=[0, 100])
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    return _style(fig, "", "Time slot", "P(X > C)", height=340)


def reliability_chart(model: pd.DataFrame, base: pd.DataFrame) -> go.Figure:
    """Calibration: forecast probability against how often overload actually happened."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line={"color": MUTED, "dash": "dot", "width": 1.5},
                    name="Perfect calibration", hoverinfo="skip")
    for frame, name, color, symbol in ((base, "Per-slot average", BLUE_SOFT, "circle"),
                                       (model, "Regression model", INK, "diamond")):
        fig.add_scatter(x=100 * frame["mean_forecast"], y=100 * frame["observed_rate"], mode="lines+markers",
                        name=name, line={"color": color, "width": 2}, marker={"size": 10, "symbol": symbol},
                        customdata=frame["n"], hovertemplate=name + "<br>forecast %{x:.0f}%<br>observed %{y:.0f}%"
                                                             "<br>%{customdata} slot forecasts<extra></extra>")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return _style(fig, "", "Forecast P(X > C)", "Observed overload frequency", height=360)


def pit_histogram(counts: list[int]) -> go.Figure:
    """Probability integral transform values: a flat histogram means calibrated forecasts."""
    edges = np.linspace(0, 1, len(counts) + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    expected = sum(counts) / len(counts)
    fig = go.Figure(go.Bar(x=mids, y=counts, width=0.9 / len(counts), marker_color=BLUE_SOFT,
                           hovertemplate="PIT %{x:.2f}<br>%{y} forecasts<extra></extra>", name="Forecasts"))
    fig.add_hline(y=expected, line_color=INK, line_dash="dash", line_width=1.5,
                  annotation_text="flat = calibrated", annotation_position="top right")
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(showlegend=False)
    return _style(fig, "", "PIT value", "Number of slot forecasts", height=320)


def daily_error_comparison(days: pd.DataFrame) -> go.Figure:
    """Each test day's mean absolute error, model against the per-slot average (below the line = model wins)."""
    hi = float(max(days["mae_base"].max(), days["mae_model"].max())) * 1.08
    fig = go.Figure()
    fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", line={"color": MUTED, "dash": "dot", "width": 1.5},
                    name="Equal error", hoverinfo="skip")
    others = sorted(set(days["day_type"]) - {"Regular"})
    styles = [("Regular", TEAL, "circle")] + [(t, c, m) for t, c, m in zip(
        others, (RISK_COLORS["MEDIUM"], INK, MUTED), ("diamond", "square", "triangle-up"))]
    for label, color, symbol in styles:
        part = days[days["day_type"] == label]
        if part.empty:
            continue
        fig.add_scatter(x=part["mae_base"], y=part["mae_model"], mode="markers", name=f"{label} days",
                        marker={"size": 9, "color": color, "symbol": symbol, "line": {"color": "#FFFFFF", "width": 1}},
                        customdata=np.stack([part["date"].dt.strftime("%d %b"), part["day"]], axis=-1),
                        hovertemplate="%{customdata[0]} (%{customdata[1]})<br>baseline MAE %{x:.2f}"
                                      "<br>model MAE %{y:.2f}<extra></extra>")
    fig.update_xaxes(range=[0, hi])
    fig.update_yaxes(range=[0, hi])
    return _style(fig, "", "Per-slot average: daily MAE", "Regression model: daily MAE", height=360)


def backtest_surface_3d(slots: pd.DataFrame) -> go.Figure:
    """Actual counts (surface) against the one-day-ahead forecasts (markers) for every test day."""
    actual = slots.pivot(index="date", columns="period", values="actual").sort_index()
    pred = slots.pivot(index="date", columns="period", values="mu_model").reindex(actual.index)[actual.columns]
    periods, dates = list(actual.columns), [f"{d:%d %b}" for d in actual.index]
    xi, yi = np.meshgrid(range(len(periods)), range(len(dates)))
    fig = go.Figure(go.Surface(z=actual.to_numpy(float), x=list(range(len(periods))), y=list(range(len(dates))),
                               colorscale=SEQUENTIAL, opacity=0.85, showscale=False, name="Actual",
                               hovertemplate="actual %{z:.0f}<extra></extra>"))
    fig.add_scatter3d(x=xi.ravel(), y=yi.ravel(), z=pred.to_numpy(float).ravel(), mode="markers",
                      marker={"size": 2.6, "color": RISK_COLORS["MEDIUM"]}, name="Forecast (one day ahead)",
                      customdata=np.stack([np.tile(periods, len(dates)), np.repeat(dates, len(periods))], axis=-1),
                      hovertemplate="%{customdata[1]}, %{customdata[0]}<br>forecast %{z:.1f}<extra></extra>")
    step = max(1, len(dates) // 8)
    return _style3d(fig, "Time slot", "Test day", "Arrivals",
                    x={"tickvals": list(range(len(periods))), "ticktext": [p[:5] for p in periods]},
                    y={"tickvals": list(range(0, len(dates), step)), "ticktext": dates[::step]})


def plan_bars(periods: list[str], probabilities, risks: list[str], best: str | None = None,
              low: float = 0.2, high: float = 0.5) -> go.Figure:
    """Student view: chance of overcrowding per time slot, in plain words and percentages."""
    words = {"LOW": "Usually fine", "MEDIUM": "Sometimes crowded", "HIGH": "Often overcrowded"}
    fig = go.Figure()
    probs = 100 * np.asarray(probabilities, dtype=float)
    labels = [p.replace("-", "–") for p in periods]
    for risk, color in RISK_COLORS.items():
        idx = [i for i, r in enumerate(risks) if r == risk]
        if not idx:
            continue
        fig.add_bar(x=[labels[i] for i in idx], y=probs[idx], name=words[risk], marker_color=color,
                    marker_line={"color": [INK if periods[i] == best else color for i in idx],
                                 "width": [3 if periods[i] == best else 0 for i in idx]},
                    text=[f"{v:.0f}%" if v >= 1 else "<1%" for v in probs[idx]], textposition="outside",
                    textfont={"color": INK},
                    hovertemplate="%{x}<br>chance of overcrowding: %{y:.0f}%<extra></extra>")
    if best in periods:
        i = periods.index(best)
        fig.add_annotation(x=labels[i], y=probs[i] + 13, text="<b>quietest</b>", showarrow=False,
                           font={"color": RISK_COLORS["LOW"], "size": 13})
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    fig.update_yaxes(range=[0, 110], ticksuffix="%", showticklabels=False)
    fig.update_layout(dragmode=False, bargap=0.3)
    return _style(fig, "", "", "Chance of overcrowding", height=360)
