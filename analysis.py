"""
analysis.py
-----------
Runs the research pipeline on a cleaned dataset:

    Exploratory analysis -> Estimate lambda -> Mean vs variance -> Goodness of fit
    -> P(X > C) -> Risk classification -> Operational recommendations

Unit of analysis
    The primary unit is the 15-minute interval (e.g. '13:00-13:15'), pooled across
    days. Optionally, the four 15-minute intervals of an hour can be pooled
    ('Hourly period'); lambda is then the mean count PER 15-MINUTE INTERVAL within
    that hour, so the capacity C keeps the same meaning in both views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from goodness_of_fit import chi_square_gof, dispersion_test
from poisson_model import (classify_risk, estimate_lambda, lambda_confidence_interval,
                           overload_probability, required_capacity, sample_variance)
from utils import EmptyDatasetError, validate_alpha, validate_capacity, validate_thresholds

GRANULARITIES = {"15-minute interval": "time_interval", "Hourly period": "hour_block"}
DEMAND_METHODS = {
    "tertile": "Tertiles (equal-frequency split of λ̂ values)",
    "zscore": "Mean ± 0.5 SD of λ̂ values (z-score bands)",
}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_dataset(df: pd.DataFrame, start_date=None, end_date=None, days=None) -> pd.DataFrame:
    """Restrict the cleaned dataset to a date range and/or a set of weekdays."""
    mask = pd.Series(True, index=df.index)
    if start_date is not None:
        mask &= df["date"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= df["date"] <= pd.Timestamp(end_date)
    if days:
        mask &= df["day"].isin(days)
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Descriptive statistics and lambda estimation per period
# ---------------------------------------------------------------------------
def summarize_by_group(df: pd.DataFrame, group_col: str = "time_interval", alpha: float = 0.05) -> pd.DataFrame:
    """Mean, variance, SD, lambda_hat (+ CI) and dispersion statistics for every period."""
    if df is None or df.empty:
        raise EmptyDatasetError("No observations available for the selected filters.")
    if group_col not in df.columns:
        raise ValueError(f"Unknown grouping column '{group_col}'.")
    alpha = validate_alpha(alpha)

    rows = []
    for period, values in df.groupby(group_col, sort=True)["arrivals"]:
        x = values.to_numpy(dtype=float)
        n, total = int(x.size), float(x.sum())
        lam = estimate_lambda(x)                     # lambda_hat = sum(x) / n
        variance = sample_variance(x)                # s^2 with (n - 1) denominator
        ci_low, ci_high = lambda_confidence_interval(total, n, 1 - alpha)
        dispersion = dispersion_test(x, alpha)
        rows.append({
            "period": period,
            "n": n,
            "total_arrivals": int(total),
            "mean": float(x.mean()),
            "variance": variance,
            "std_dev": math.sqrt(variance) if not math.isnan(variance) else float("nan"),
            "lambda_hat": lam,
            "lambda_ci_low": ci_low,
            "lambda_ci_high": ci_high,
            "var_mean_ratio": variance / lam if lam > 0 else float("nan"),
            "mean_var_ratio": lam / variance if variance and variance > 0 else float("nan"),
            "dispersion_p": dispersion["p_value"],
            "dispersion": dispersion["label"],
            "min": int(x.min()),
            "max": int(x.max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demand levels (Section 8 - algorithmic, not arbitrary)
# ---------------------------------------------------------------------------
def classify_demand_levels(lambdas: pd.Series, method: str = "tertile") -> tuple[pd.Series, dict]:
    """Label each period Low / Medium / High demand from its lambda_hat.

    tertile : cut points are the 33.3rd and 66.7th percentiles of the lambda_hat values,
              giving three equal-sized groups (relative ranking).
    zscore  : cut points are mean(lambda_hat) -/+ 0.5 * SD(lambda_hat); periods more than
              half a standard deviation below/above the average are Low/High.
    """
    lambdas = pd.Series(lambdas, dtype=float)
    if lambdas.empty:
        raise EmptyDatasetError("No periods to classify.")
    if method == "tertile":
        lower, upper = (float(q) for q in np.quantile(lambdas, [1 / 3, 2 / 3]))
        levels = np.where(lambdas <= lower, "Low", np.where(lambdas > upper, "High", "Medium"))
        rule = f"Low if λ̂ ≤ {lower:.2f} (33.3rd pct), High if λ̂ > {upper:.2f} (66.7th pct), otherwise Medium."
    elif method == "zscore":
        mean, sd = float(lambdas.mean()), float(lambdas.std(ddof=1)) if len(lambdas) > 1 else 0.0
        lower, upper = mean - 0.5 * sd, mean + 0.5 * sd
        if sd == 0 or math.isnan(sd):
            levels = np.full(len(lambdas), "Medium")
        else:
            levels = np.where(lambdas < lower, "Low", np.where(lambdas > upper, "High", "Medium"))
        rule = (f"Low if λ̂ < {lower:.2f}, High if λ̂ > {upper:.2f} "
                f"(mean {mean:.2f} ± 0.5 × SD {sd:.2f}), otherwise Medium.")
    else:
        raise ValueError(f"Unknown demand classification method '{method}'.")
    return pd.Series(levels, index=lambdas.index), {"method": method, "lower": lower, "upper": upper, "rule": rule}


# ---------------------------------------------------------------------------
# Overload probability and risk (Sections 9-10)
# ---------------------------------------------------------------------------
def add_overload_columns(summary: pd.DataFrame, capacity: int, low: float, high: float) -> pd.DataFrame:
    """Add P(X > C) (point estimate and at the lambda CI bounds) and the risk level."""
    capacity = validate_capacity(capacity)
    low, high = validate_thresholds(low, high)
    out = summary.copy()
    out["capacity"] = capacity
    out["overload_prob"] = out["lambda_hat"].map(lambda lam: overload_probability(lam, capacity))
    out["overload_prob_low"] = out["lambda_ci_low"].map(lambda lam: overload_probability(lam, capacity))
    out["overload_prob_high"] = out["lambda_ci_high"].map(lambda lam: overload_probability(lam, capacity))
    out["risk"] = out["overload_prob"].map(lambda p: classify_risk(p, low, high))
    return out


# ---------------------------------------------------------------------------
# Goodness of fit for every period (Section 11)
# ---------------------------------------------------------------------------
def run_gof_for_groups(df: pd.DataFrame, group_col: str = "time_interval", alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for period, values in df.groupby(group_col, sort=True)["arrivals"]:
        result = chi_square_gof(values.to_numpy(), alpha)
        rows.append({"period": period, "n": result.n, "lambda_hat": result.lambda_hat,
                     "categories": len(result.table), "chi_square": result.chi_square, "df": result.df,
                     "p_value": result.p_value, "decision": result.decision, "valid": result.valid})
    return pd.DataFrame(rows)


def gof_rejection_summary(gof_table: pd.DataFrame, alpha: float) -> dict:
    """How many periods reject H0, compared with what chance alone would produce.

    If H0 were true in all m periods, the number of rejections ~ Binomial(m, alpha).
    `binom_p` = P(at least this many rejections | H0 true everywhere).
    """
    valid = gof_table[gof_table["valid"]] if not gof_table.empty else gof_table
    m = int(len(valid))
    rejections = int((valid["decision"] == "Reject H0").sum()) if m else 0
    binom_p = float(stats.binom.sf(rejections - 1, m, alpha)) if m else float("nan")
    return {"tested": m, "rejections": rejections, "expected_by_chance": alpha * m,
            "binom_p": binom_p, "excessive": bool(m and binom_p <= alpha)}


# ---------------------------------------------------------------------------
# Recommendations (Section 13, part 5)
# ---------------------------------------------------------------------------
def contiguous_windows(periods: list[str], flags: list[bool]) -> list[list[str]]:
    """Group consecutive flagged periods ('HH:MM-HH:MM' labels, time-ordered) into windows."""
    windows, current = [], []
    for label, flagged in zip(periods, flags):
        if flagged and current and current[-1].split("-")[1] == label.split("-")[0]:
            current.append(label)
        elif flagged:
            if current:
                windows.append(current)
            current = [label]
        else:
            if current:
                windows.append(current)
            current = []
    if current:
        windows.append(current)
    return windows


def window_label(window: list[str]) -> str:
    return f"{window[0].split('-')[0]}–{window[-1].split('-')[1]}"


def _prob_range(probabilities: pd.Series) -> str:
    low, high = 100 * probabilities.min(), 100 * probabilities.max()
    return f"{low:.1f}%" if round(low, 1) == round(high, 1) else f"{low:.1f}% to {high:.1f}%"


def generate_recommendations(table: pd.DataFrame, gof_summary: dict, capacity: int,
                             low: float, high: float) -> list[str]:
    """Plain-language, model-based operational suggestions."""
    recs = []
    table = table.sort_values("period").reset_index(drop=True)
    periods = table["period"].tolist()
    peak = table.loc[table["lambda_hat"].idxmax()]

    recs.append(
        f"**Highest-demand period: {peak['period']}.** It has the highest estimated arrival rate "
        f"(λ̂ = {peak['lambda_hat']:.1f} arrivals per 15-minute interval, "
        f"{100 * (1 - gof_summary.get('alpha', 0.05)):.0f}% CI {peak['lambda_ci_low']:.1f}–{peak['lambda_ci_high']:.1f}). "
        f"The model-estimated probability of exceeding the configured capacity of {capacity} arrivals "
        f"in this period is {100 * peak['overload_prob']:.1f}% ({peak['risk']} risk).")

    high_windows = contiguous_windows(periods, (table["risk"] == "HIGH").tolist())
    for window in high_windows:
        rows = table[table["period"].isin(window)]
        recs.append(
            f"**{window_label(window)} — HIGH overload risk.** Estimated P(X > {capacity}): "
            f"{_prob_range(rows['overload_prob'])} across {len(window)} period(s). "
            "Additional service staff or an extra counter may be considered during this window.")
    if not high_windows:
        recs.append(f"No period exceeds the HIGH-risk threshold ({100 * high:.0f}%) at a capacity of {capacity}.")

    for window in contiguous_windows(periods, (table["risk"] == "MEDIUM").tolist()):
        rows = table[table["period"].isin(window)]
        recs.append(
            f"**{window_label(window)} — MEDIUM overload risk** (estimated P(X > {capacity}): "
            f"{_prob_range(rows['overload_prob'])}). "
            "Pre-preparing popular items or keeping a standby staff member available may help.")

    needed = required_capacity(float(peak["lambda_hat"]), low)
    if needed > capacity:
        recs.append(
            f"**Capacity planning:** to keep the estimated overload probability below {100 * low:.0f}% in "
            f"every period, the model suggests handling at least **{needed} arrivals per 15-minute interval** "
            f"in the busiest period (currently configured: {capacity}).")
    else:
        recs.append(f"**Capacity planning:** the configured capacity ({capacity}) keeps the estimated overload "
                    f"probability below {100 * low:.0f}% in every period.")

    if "demand_level" in table.columns:
        low_windows = contiguous_windows(periods, (table["demand_level"] == "Low").tolist())
        low_windows = sorted(low_windows, key=len, reverse=True)[:3]
        if low_windows:
            recs.append("**Low-demand windows** such as " + ", ".join(window_label(w) for w in low_windows)
                        + " may be suitable for restocking, cleaning and staff breaks.")

    if gof_summary.get("tested"):
        g = gof_summary
        if g["excessive"]:
            recs.append(
                f"**Model-fit caution:** the Poisson fit was rejected in {g['rejections']} of {g['tested']} periods, "
                f"more than the ≈{g['expected_by_chance']:.1f} expected by chance (binomial p = {g['binom_p']:.3f}). "
                "Probabilities for the rejected periods should be interpreted with care.")
        else:
            recs.append(
                f"**Model fit:** the Poisson fit was rejected in {g['rejections']} of {g['tested']} periods, which is "
                f"in line with the ≈{g['expected_by_chance']:.1f} false rejections expected by chance at this α.")

    recs.append("_These are model-based estimates computed from operational arrival records. They describe expected "
                "behaviour under the Poisson model and do not guarantee future arrivals._")
    return recs


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    table: pd.DataFrame              # one row per period: statistics, demand level, P(X > C), risk
    gof_table: pd.DataFrame          # chi-square GOF result per period
    gof_summary: dict
    demand_info: dict
    recommendations: list[str]
    kpis: dict = field(default_factory=dict)


def run_pipeline(df: pd.DataFrame, capacity: int, alpha: float = 0.05, low: float = 0.20, high: float = 0.50,
                 group_col: str = "time_interval", demand_method: str = "tertile") -> PipelineResult:
    """Run estimation -> dispersion -> GOF -> overload -> risk -> recommendations."""
    if df is None or df.empty:
        raise EmptyDatasetError("No observations available for the selected filters.")
    capacity = validate_capacity(capacity)
    alpha = validate_alpha(alpha)
    low, high = validate_thresholds(low, high)

    table = summarize_by_group(df, group_col, alpha)
    table["demand_level"], demand_info = classify_demand_levels(table["lambda_hat"], demand_method)
    table = add_overload_columns(table, capacity, low, high)

    gof_table = run_gof_for_groups(df, group_col, alpha)
    gof_summary = gof_rejection_summary(gof_table, alpha) | {"alpha": alpha}
    table = table.merge(gof_table[["period", "p_value", "decision"]].rename(
        columns={"p_value": "gof_p_value", "decision": "gof_decision"}), on="period", how="left")

    recommendations = generate_recommendations(table, gof_summary, capacity, low, high)

    max_row = df.loc[df["arrivals"].idxmax()]
    peak = table.loc[table["lambda_hat"].idxmax()]
    worst = table.loc[table["overload_prob"].idxmax()]
    kpis = {
        "total_observations": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "n_periods": int(len(table)),
        "avg_arrivals": float(df["arrivals"].mean()),
        "max_arrivals": int(max_row["arrivals"]),
        "max_arrivals_when": f"{max_row['date']:%d %b %Y}, {max_row['time_interval']}",
        "peak_period": peak["period"],
        "peak_lambda": float(peak["lambda_hat"]),
        "max_overload_period": worst["period"],
        "max_overload_prob": float(worst["overload_prob"]),
        "risk_counts": table["risk"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0).to_dict(),
    }
    return PipelineResult(table, gof_table, gof_summary, demand_info, recommendations, kpis)
