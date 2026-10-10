"""
poisson_model.py
----------------
The Poisson statistical model: lambda estimation, PMF/CDF, overload probability
and risk classification.

Every probability is implemented twice:
  * a *manual* version that follows the textbook formula line by line
    (for academic explanation), and
  * a SciPy version (numerically more precise for very small tail probabilities),
    which the dashboard uses for the displayed numbers.
The automated tests check that both versions agree.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from utils import validate_capacity, validate_thresholds

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def _validate_lambda(lam: float) -> float:
    if lam is None or isinstance(lam, bool):
        raise ValueError("lambda must be a non-negative number.")
    lam = float(lam)
    if math.isnan(lam) or math.isinf(lam) or lam < 0:
        raise ValueError(f"lambda must be a finite, non-negative number (got {lam}).")
    return lam


def _validate_count(x: int) -> int:
    if isinstance(x, bool) or float(x) != int(x) or x < 0:
        raise ValueError(f"x must be a non-negative whole number (got {x}).")
    return int(x)


def to_count_array(arrivals: Iterable) -> np.ndarray:
    """Convert any sequence of counts to a float array, dropping missing values."""
    series = pd.to_numeric(pd.Series(list(arrivals), dtype="object"), errors="coerce")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate from an empty set of observations.")
    if np.any(values < 0):
        raise ValueError("Arrival counts cannot be negative.")
    return values


# ---------------------------------------------------------------------------
# Estimation:  lambda_hat = sum(x) / n
# ---------------------------------------------------------------------------
def estimate_lambda(arrivals: Iterable) -> float:
    """Maximum-likelihood estimate of lambda: total arrivals / number of observations."""
    values = to_count_array(arrivals)
    return float(values.sum() / values.size)


def sample_mean(arrivals: Iterable) -> float:
    values = to_count_array(arrivals)
    return float(values.mean())


def sample_variance(arrivals: Iterable) -> float:
    """Unbiased sample variance  s^2 = sum (x - xbar)^2 / (n - 1). NaN if n < 2."""
    values = to_count_array(arrivals)
    if values.size < 2:
        return float("nan")
    return float(values.var(ddof=1))


def lambda_confidence_interval(total_arrivals: float, n_observations: int,
                               confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) confidence interval for a Poisson rate.

    If S = sum of n Poisson(lambda) counts then S ~ Poisson(n * lambda) and
        lower = chi2_{a/2}(2S) / (2n),   upper = chi2_{1-a/2}(2S + 2) / (2n)
    """
    if n_observations <= 0:
        raise ValueError("Need at least one observation for a confidence interval.")
    if total_arrivals < 0:
        raise ValueError("Total arrivals cannot be negative.")
    a = 1.0 - confidence
    lower = 0.0 if total_arrivals == 0 else stats.chi2.ppf(a / 2, 2 * total_arrivals) / (2 * n_observations)
    upper = stats.chi2.ppf(1 - a / 2, 2 * total_arrivals + 2) / (2 * n_observations)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Manual (textbook) Poisson formulas
# ---------------------------------------------------------------------------
def poisson_pmf_manual(x: int, lam: float) -> float:
    """P(X = x) = e^(-lambda) * lambda^x / x!

    Evaluated in log form, log P = -lambda + x*ln(lambda) - ln(x!), which is the
    same formula but avoids overflow of lambda^x and x! for larger values.
    """
    x, lam = _validate_count(x), _validate_lambda(lam)
    if lam == 0:
        return 1.0 if x == 0 else 0.0
    return math.exp(-lam + x * math.log(lam) - math.lgamma(x + 1))


def poisson_cdf_manual(c: int, lam: float) -> float:
    """P(X <= c) = sum_{x=0}^{c} e^(-lambda) lambda^x / x!"""
    c = _validate_count(c)
    return min(1.0, sum(poisson_pmf_manual(x, lam) for x in range(c + 1)))


def overload_probability_manual(lam: float, capacity: int) -> float:
    """P(X > C) = 1 - P(X <= C), computed with the manual formula."""
    capacity = validate_capacity(capacity)
    return max(0.0, 1.0 - poisson_cdf_manual(capacity, lam))


# ---------------------------------------------------------------------------
# SciPy versions (used for displayed results)
# ---------------------------------------------------------------------------
def poisson_pmf(x: int, lam: float) -> float:
    return float(stats.poisson.pmf(_validate_count(x), _validate_lambda(lam)))


def overload_probability(lam: float, capacity: int) -> float:
    """P(X > C) using SciPy's survival function sf(C) = 1 - cdf(C)."""
    lam, capacity = _validate_lambda(lam), validate_capacity(capacity)
    return float(stats.poisson.sf(capacity, lam))


def overload_breakdown(lam: float, capacity: int) -> pd.DataFrame:
    """Worked table x = 0..C with P(X = x) and the running total P(X <= x)."""
    capacity = validate_capacity(capacity)
    rows, cumulative = [], 0.0
    for x in range(capacity + 1):
        p = poisson_pmf_manual(x, lam)
        cumulative += p
        rows.append({"x": x, "P(X = x)": p, "P(X ≤ x)": min(cumulative, 1.0)})
    return pd.DataFrame(rows)


def simulate_overload_frequency(lam: float, capacity: int, n_days: int,
                                seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo check of P(X > C): simulate `n_days` lunch services of one period.

    Returns (draws, running_frequency) where running_frequency[k] is the share of the
    first k + 1 simulated days that overflowed. By the law of large numbers it
    converges to the model value poisson.sf(C, lambda).
    """
    lam, capacity = _validate_lambda(lam), validate_capacity(capacity)
    if isinstance(n_days, bool) or int(n_days) < 1:
        raise ValueError(f"Number of simulated days must be at least 1 (got {n_days}).")
    draws = np.random.default_rng(seed).poisson(lam, int(n_days))
    running = np.cumsum(draws > capacity) / np.arange(1, draws.size + 1)
    return draws, running


def required_capacity(lam: float, max_overload_probability: float) -> int:
    """Smallest capacity C (>= 1) such that P(X > C) < max_overload_probability."""
    lam = _validate_lambda(lam)
    if not 0.0 < max_overload_probability < 1.0:
        raise ValueError("Target overload probability must be between 0 and 1.")
    c = max(1, int(stats.poisson.ppf(1.0 - max_overload_probability, lam)))
    while stats.poisson.sf(c, lam) >= max_overload_probability:
        c += 1
    return c


# ---------------------------------------------------------------------------
# Decision rule (project-defined thresholds, NOT universal standards)
# ---------------------------------------------------------------------------
def classify_risk(probability: float, low_threshold: float = 0.20, high_threshold: float = 0.50) -> str:
    """LOW if p < low;  HIGH if p > high;  MEDIUM otherwise (low <= p <= high)."""
    low_threshold, high_threshold = validate_thresholds(low_threshold, high_threshold)
    if probability is None or math.isnan(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1 (got {probability}).")
    if probability < low_threshold:
        return "LOW"
    if probability > high_threshold:
        return "HIGH"
    return "MEDIUM"
