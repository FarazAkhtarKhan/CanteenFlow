"""
goodness_of_fit.py
------------------
Statistical evidence for (or against) the Poisson assumption.

1. Chi-square goodness-of-fit test
   H0: the observed arrival counts follow a Poisson distribution.
   H1: the observed arrival counts do not follow a Poisson distribution.

   chi^2 = sum (O_i - E_i)^2 / E_i,  df = (number of categories) - 1 - 1
   (one extra degree of freedom is lost because lambda is estimated from the data).

   Handling low expected frequencies (validity of the chi-square approximation)
   ---------------------------------------------------------------------------
   We start from single count values 0, 1, 2, ... covering the whole Poisson
   support, and merge ADJACENT values from left to right until every merged
   category has expected frequency E_i >= 5 (Cochran's rule). Any leftover
   right-tail category with E_i < 5 is merged into the previous category.
   The first category is therefore "X <= a" and the last "X >= b", so the
   categories cover every possible count and the expected probabilities sum to 1.
   The binning depends only on lambda_hat and n, never on the observed counts.
   If fewer than 3 categories remain (df < 1) the test is reported as not valid.

2. Index-of-dispersion test (supporting evidence)
   D = (n - 1) s^2 / xbar  ~  chi^2(n - 1) under H0 (Poisson), two-sided.
   Checks whether Var/Mean differs from 1 by more than chance would explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from poisson_model import to_count_array
from utils import validate_alpha

MIN_EXPECTED_FREQUENCY = 5.0


@dataclass
class GOFResult:
    lambda_hat: float
    n: int
    alpha: float
    valid: bool
    chi_square: float = float("nan")
    df: int = 0
    p_value: float = float("nan")
    reject_h0: bool | None = None
    message: str = ""
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    merge_notes: list[str] = field(default_factory=list)

    @property
    def decision(self) -> str:
        if not self.valid:
            return "Test not valid"
        return "Reject H0" if self.reject_h0 else "Fail to reject H0"


def _category_label(lower: int, upper: float) -> str:
    if not np.isfinite(upper):
        return f"≥ {lower}"
    if lower == upper:
        return str(lower)
    if lower == 0:
        return f"≤ {int(upper)}"
    return f"{lower}–{int(upper)}"


def build_frequency_table(observations, lam: float, min_expected: float = MIN_EXPECTED_FREQUENCY):
    """Observed vs expected frequencies with low-expected categories merged.

    Returns (table, merge_notes). Table columns:
        category, lower, upper, observed, expected, probability
    """
    obs = to_count_array(observations).astype(int)
    n = obs.size
    if lam <= 0:
        raise ValueError("lambda must be positive to build expected frequencies.")

    # Single-value categories 0..K, where P(X >= K) is negligible; the last one is open-ended.
    k_max = int(max(obs.max(), stats.poisson.isf(1e-12, lam))) + 1
    singles = [(k, k) for k in range(k_max)] + [(k_max, np.inf)]

    def probability(lower, upper):
        if np.isinf(upper):
            return float(stats.poisson.sf(lower - 1, lam))
        return float(stats.poisson.cdf(upper, lam) - stats.poisson.cdf(lower - 1, lam))

    # Greedy left-to-right merging until expected >= min_expected
    groups, current = [], None
    for lower, upper in singles:
        current = (lower, upper) if current is None else (current[0], upper)
        if n * probability(*current) >= min_expected:
            groups.append(current)
            current = None
    notes = []
    if current is not None:  # leftover right tail with expected < min_expected
        if groups:
            groups[-1] = (groups[-1][0], current[1])
            notes.append("The right-tail remainder had expected frequency < "
                         f"{min_expected:g} and was merged into the previous category.")
        else:
            groups.append(current)

    rows = []
    for lower, upper in groups:
        p = probability(lower, upper)
        observed = int(np.sum((obs >= lower) & (obs <= upper)))
        rows.append({"category": _category_label(lower, upper), "lower": lower,
                     "upper": upper, "observed": observed, "probability": p, "expected": n * p})
    table = pd.DataFrame(rows)

    merged = [r["category"] for r in rows if r["lower"] != r["upper"]]
    notes.insert(0, f"Single count values were merged with neighbours until each category had expected "
                    f"frequency ≥ {min_expected:g}. Merged categories: {', '.join(merged) if merged else 'none'}.")
    return table, notes


def chi_square_gof(observations, alpha: float = 0.05,
                   min_expected: float = MIN_EXPECTED_FREQUENCY) -> GOFResult:
    """Chi-square goodness-of-fit test of a Poisson(lambda_hat) model."""
    alpha = validate_alpha(alpha)
    obs = to_count_array(observations)
    n = int(obs.size)
    lam = float(obs.mean())

    if lam == 0:
        return GOFResult(lam, n, alpha, valid=False,
                         message="All observed counts are zero; the Poisson fit cannot be tested.")

    table, notes = build_frequency_table(obs, lam, min_expected)
    k = len(table)
    df = k - 1 - 1  # categories - 1 - (number of estimated parameters)
    if df < 1:
        return GOFResult(lam, n, alpha, valid=False, table=table, merge_notes=notes,
                         message=(f"Only {k} category(ies) have expected frequency ≥ {min_expected:g}; "
                                  "at least 3 are needed (df ≥ 1). More observations are required."))

    table["contribution"] = (table["observed"] - table["expected"]) ** 2 / table["expected"]
    chi_sq = float(table["contribution"].sum())
    p_value = float(stats.chi2.sf(chi_sq, df))
    reject = p_value <= alpha
    if reject:
        message = (f"p-value = {p_value:.4f} ≤ α = {alpha:g}: reject H0. There is evidence that the "
                   "observed counts do not fit a Poisson distribution well.")
    else:
        message = (f"p-value = {p_value:.4f} > α = {alpha:g}: fail to reject H0. The Poisson model is "
                   "reasonably consistent with the observed data (this does not prove it is Poisson).")
    return GOFResult(lam, n, alpha, valid=True, chi_square=chi_sq, df=df, p_value=p_value,
                     reject_h0=reject, message=message, table=table, merge_notes=notes)


def dispersion_test(observations, alpha: float = 0.05) -> dict:
    """Two-sided index-of-dispersion test of Var = Mean (Poisson property)."""
    alpha = validate_alpha(alpha)
    obs = to_count_array(observations)
    n = obs.size
    mean = float(obs.mean())
    if n < 2 or mean == 0:
        return {"index": float("nan"), "statistic": float("nan"), "df": max(n - 1, 0),
                "p_value": float("nan"), "label": "Not enough data"}
    variance = float(obs.var(ddof=1))
    index = variance / mean
    statistic = (n - 1) * index
    df = n - 1
    p_value = float(min(1.0, 2 * min(stats.chi2.cdf(statistic, df), stats.chi2.sf(statistic, df))))
    if p_value > alpha:
        label = "Consistent with Poisson"
    else:
        label = "Overdispersed" if index > 1 else "Underdispersed"
    return {"index": index, "statistic": statistic, "df": df, "p_value": p_value, "label": label}
