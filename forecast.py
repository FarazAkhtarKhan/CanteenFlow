"""
forecast.py
-----------
Peak-time prediction and forecast validation (research gaps G7-G11).

Every method here stays inside the PAMAT501 syllabus:

    G7  Day-specific rates     multiple linear regression on log counts with slot, weekday and
                               day-type dummies (Module 3), t-tests on the coefficients and
                               ANOVA / Kruskal-Wallis / t / Mann-Whitney tests of the day effects (Module 5)
    G8  Peak-time probability  Monte Carlo over Poisson counts: P(slot t is the peak),
                               P(at least one overload) (Modules 1-2)
    G9  Same-day updating      Bayes theorem with a Gamma prior on the day's busyness factor W
                               and Poisson counts -> Gamma posterior (Modules 1-2)
    G10 Overdispersion         Poisson-Gamma mixture via the theorem of total probability:
                               Var(X) = lambda + v lambda^2; Pearson chi-square test (Modules 1, 2, 5)
    G11 Validation             rolling one-day-ahead backtest: Brier score and skill score,
                               proportion z-tests (Module 4), paired t and sign tests,
                               Kolmogorov-Smirnov and chi-square tests of the PIT (Module 5),
                               correlation of predicted and actual counts (Module 3)
    Extras                     Spearman rank correlation between consecutive slots (Module 3),
                               runs test of day-to-day residual signs (Module 5)

Model
=====
For day d and slot t, X_dt | W_d ~ Poisson(W_d * mu_dt), where
    log mu_dt = b0 + slot_t + weekday_d + daytype_d        (fitted by least squares on log(x + 0.5))
and W_d ~ Gamma(shape 1/v, rate 1/v) is the day's busyness factor (mean 1, variance v).
Integrating W out gives X_dt ~ NegativeBinomial with mean mu and variance mu + v mu^2;
with v = 0 this is the plain Poisson model used in the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from utils import DAY_NAMES

DAY_TYPE_COLUMN = "day_type"
REGULAR = "Regular"
LOG_OFFSET = 0.5            # log(x + 0.5) keeps zero counts finite
MIN_VARIANCE = 1e-6         # below this the day factor is treated as constant (pure Poisson)
DEFAULT_TRAIN_DAYS = 30
DEFAULT_SIMULATIONS = 20_000


# ===========================================================================
# Distributions (G10)
# ===========================================================================
def _nb_params(lam, var_w: float):
    """scipy nbinom (n, p) with mean lam and variance lam + var_w * lam^2."""
    n = 1.0 / var_w
    return n, n / (n + np.asarray(lam, dtype=float))


def overload_probability_mixed(lam, capacity: int, var_w: float = 0.0):
    """P(X > C) when X ~ Poisson(W * lam) and W ~ Gamma(mean 1, variance var_w).

    By the theorem of total probability P(X = x) = ∫ Poisson(x; w lam) Gamma(w) dw, which is the
    negative binomial. With var_w = 0 this is the ordinary Poisson tail used elsewhere.
    """
    if var_w <= MIN_VARIANCE:
        return stats.poisson.sf(capacity, lam)
    n, p = _nb_params(lam, var_w)
    return stats.nbinom.sf(capacity, n, p)


def predictive_cdf(x, lam, var_w: float = 0.0):
    """P(X <= x) under the Poisson (var_w = 0) or Poisson-Gamma model."""
    if var_w <= MIN_VARIANCE:
        return stats.poisson.cdf(x, lam)
    n, p = _nb_params(lam, var_w)
    return stats.nbinom.cdf(x, n, p)


def predictive_interval(lam, var_w: float = 0.0, level: float = 0.95):
    """Equal-tailed prediction interval (lower, upper) for a count."""
    tail = (1 - level) / 2
    if var_w <= MIN_VARIANCE:
        return stats.poisson.ppf(tail, lam), stats.poisson.ppf(1 - tail, lam)
    n, p = _nb_params(lam, var_w)
    return stats.nbinom.ppf(tail, n, p), stats.nbinom.ppf(1 - tail, n, p)


# ===========================================================================
# G7: multiple linear regression for day-specific rates
# ===========================================================================
@dataclass
class RateModel:
    """Least-squares fit of log(x + 0.5) on slot, weekday and day-type dummy variables."""
    coefficients: pd.DataFrame          # term, estimate, std_error, t, p_value
    periods: list[str]
    weekdays: list[str]
    day_types: list[str]
    beta: np.ndarray
    smear: float                        # back-transformation factor mean(exp(residual))
    r_squared: float
    adj_r_squared: float
    sigma: float                        # residual standard deviation on the log scale
    n: int
    dof: int
    fitted: pd.Series = field(repr=False)  # fitted mean count for each training row (index = row index)

    def design_row(self, period: str, weekday: str, day_type: str = REGULAR) -> np.ndarray:
        row = [1.0]
        row += [1.0 if period == p else 0.0 for p in self.periods[1:]]
        row += [1.0 if weekday == w else 0.0 for w in self.weekdays[1:]]
        row += [1.0 if day_type == d else 0.0 for d in self.day_types[1:]]
        return np.array(row)

    def predict(self, weekday: str, day_type: str = REGULAR, periods: list[str] | None = None) -> pd.Series:
        """Predicted mean arrivals mu_t for every slot of a day with this weekday and day type."""
        periods = periods or self.periods
        if weekday not in self.weekdays:
            raise ValueError(f"No training days fall on a {weekday}.")
        if day_type not in self.day_types:
            raise ValueError(f"No training days are of type '{day_type}'.")
        X = np.vstack([self.design_row(p, weekday, day_type) for p in periods])
        return pd.Series(_back_transform(X @ self.beta, self.smear), index=periods, name="mu")


def _back_transform(log_values, smear: float):
    return np.maximum(np.exp(log_values) * smear - LOG_OFFSET, 0.05)


def _levels(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    ordered = [v for v in preferred if v in present] + sorted(v for v in present if v not in preferred)
    return ordered


def day_types_of(df: pd.DataFrame) -> pd.Series:
    if DAY_TYPE_COLUMN in df.columns:
        return df[DAY_TYPE_COLUMN].fillna(REGULAR).astype(str)
    return pd.Series(REGULAR, index=df.index)


def fit_rate_model(df: pd.DataFrame, group_col: str = "time_interval") -> RateModel:
    """Fit log(x + 0.5) = b0 + slot + weekday + day type + e by ordinary least squares.

    The first level of each factor is the reference (dropped dummy). Coefficient t-tests use
    t = estimate / SE on n - p degrees of freedom. Fitted means are back-transformed with the
    smearing factor mean(exp(e)) so that they estimate E[X] rather than the median.
    """
    if df is None or df.empty:
        raise ValueError("No observations to fit.")
    periods = sorted(df[group_col].unique())
    weekdays = _levels(df["day"], DAY_NAMES)
    types = day_types_of(df)
    day_types = _levels(types, [REGULAR])

    cols = {"Intercept": np.ones(len(df))}
    for p in periods[1:]:
        cols[f"slot {p}"] = (df[group_col] == p).to_numpy(float)
    for w in weekdays[1:]:
        cols[w] = (df["day"] == w).to_numpy(float)
    for d in day_types[1:]:
        cols[f"{d} day"] = (types == d).to_numpy(float)
    X = np.column_stack(list(cols.values()))
    y = np.log(df["arrivals"].to_numpy(float) + LOG_OFFSET)
    n, k = X.shape
    dof = n - k
    if dof < 1:
        raise ValueError("Not enough observations to fit the regression (need more rows than terms).")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    sigma2 = sse / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p_values = 2 * stats.t.sf(np.abs(t), dof)
    coef = pd.DataFrame({"term": list(cols), "estimate": beta, "std_error": se, "t": t, "p_value": p_values,
                         "multiplier": np.exp(beta)})
    r2 = 1 - sse / sst if sst > 0 else 0.0
    smear = float(np.mean(np.exp(resid)))
    fitted = pd.Series(_back_transform(X @ beta, smear), index=df.index, name="mu")
    return RateModel(coef, periods, weekdays, day_types, beta, smear, r2, 1 - (1 - r2) * (n - 1) / dof,
                     float(np.sqrt(sigma2)), n, dof, fitted)


def day_effect_tests(df: pd.DataFrame) -> dict:
    """Do weekdays (and exam/regular days) differ in total lunch arrivals?

    Uses daily totals on complete days only. Weekdays: one-way ANOVA and Kruskal-Wallis on regular
    days. Day types: Welch two-sample t-test and Mann-Whitney U test (Module 5).
    """
    n_slots = df["time_interval"].nunique()
    daily = (df.assign(_type=day_types_of(df)).groupby("date")
             .agg(total=("arrivals", "sum"), slots=("arrivals", "size"), day=("day", "first"),
                  day_type=("_type", "first")))
    daily = daily[daily["slots"] == n_slots]
    out: dict = {"n_days": int(len(daily))}
    regular = daily[daily["day_type"] == REGULAR]
    groups = [g["total"].to_numpy() for _, g in regular.groupby("day") if len(g) >= 2]
    if len(groups) >= 2:
        f_stat, f_p = stats.f_oneway(*groups)
        h_stat, h_p = stats.kruskal(*groups)
        out["weekday"] = {"anova_f": float(f_stat), "anova_p": float(f_p), "kruskal_h": float(h_stat),
                          "kruskal_p": float(h_p), "groups": len(groups),
                          "means": regular.groupby("day")["total"].mean().reindex(
                              [d for d in DAY_NAMES if d in set(regular["day"])]).to_dict()}
    special = daily[daily["day_type"] != REGULAR]
    if len(special) >= 2 and len(regular) >= 2:
        t_stat, t_p = stats.ttest_ind(special["total"], regular["total"], equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(special["total"], regular["total"], alternative="two-sided")
        out["day_type"] = {"t": float(t_stat), "t_p": float(t_p), "u": float(u_stat), "u_p": float(u_p),
                           "special_mean": float(special["total"].mean()),
                           "regular_mean": float(regular["total"].mean()),
                           "n_special": int(len(special)), "n_regular": int(len(regular))}
    return out


# ===========================================================================
# G10: day-to-day variation (overdispersion)
# ===========================================================================
def estimate_day_variance(df: pd.DataFrame, model: RateModel) -> dict:
    """Method-of-moments estimate of v = Var(W) and a Pearson chi-square test of the Poisson model.

    Under the Poisson-Gamma model E[(X - mu)^2 / mu] = 1 + v mu. The Pearson statistic
        X^2 = sum (x - mu)^2 / mu
    therefore has expectation about (n - p) + v sum(mu), which gives the moment estimator
        v_hat = (X^2 - (n - p)) / sum(mu)                  (clipped at 0),
    with n - p the residual degrees of freedom (Cameron & Trivedi, 1990). If the plain Poisson model
    holds, X^2 ~ chi-square(n - p), which is the test reported here (Module 5).
    """
    x = df["arrivals"].to_numpy(float)
    mu = model.fitted.loc[df.index].to_numpy(float)
    pearson = float(np.sum((x - mu) ** 2 / mu))
    dof = model.dof
    v = float(max((pearson - dof) / np.sum(mu), 0.0))
    return {"var_w": v, "cv_w": float(np.sqrt(v)), "pearson_chi2": pearson, "dof": dof,
            "dispersion_ratio": pearson / dof, "p_value": float(stats.chi2.sf(pearson, dof))}


# ===========================================================================
# G8: peak-time probability
# ===========================================================================
def peak_time_distribution(mu: pd.Series, capacity: int, var_w: float = 0.0,
                           n_sims: int = DEFAULT_SIMULATIONS, seed: int = 0) -> dict:
    """Simulate whole lunch periods to find how likely each slot is to be the busiest.

    Each simulated day draws W ~ Gamma(mean 1, variance var_w) and X_t ~ Poisson(W mu_t)
    independently per slot. Ties for the maximum are broken at random, so the probabilities sum to 1.
    """
    rng = np.random.default_rng(seed)
    lam = mu.to_numpy(float)
    if var_w > MIN_VARIANCE:
        w = rng.gamma(1.0 / var_w, var_w, size=(n_sims, 1))
    else:
        w = np.ones((n_sims, 1))
    counts = rng.poisson(w * lam)
    tie_break = rng.random(counts.shape) * 0.5
    peak_idx = np.argmax(counts + tie_break, axis=1)
    p_peak = np.bincount(peak_idx, minlength=len(lam)) / n_sims
    over = counts > capacity
    p_peak_s = pd.Series(p_peak, index=mu.index, name="p_peak")
    best = p_peak_s.idxmax()
    order = np.argsort(-p_peak)
    top2 = mu.index[order[:2]].tolist()
    return {
        "p_peak": p_peak_s,
        "most_likely_peak": best,
        "p_most_likely": float(p_peak_s.max()),
        "top_two": top2,
        "p_top_two": float(p_peak[order[:2]].sum()),
        "p_overload_sim": pd.Series(over.mean(axis=0), index=mu.index),
        "p_any_overload": float(over.any(axis=1).mean()),
        "expected_overloaded_slots": float(over.sum(axis=1).mean()),
        "peak_count_quantiles": np.quantile(counts.max(axis=1), [0.05, 0.5, 0.95]).tolist(),
        "n_sims": n_sims,
    }


# ===========================================================================
# G9: same-day Bayesian updating of the busyness factor
# ===========================================================================
def update_day_factor(mu_observed, observed, var_w: float) -> tuple[float, float]:
    """Gamma posterior (shape, rate) of today's busyness factor W.

    Prior W ~ Gamma(a, a) with a = 1/var_w (mean 1). Counts X_t | W ~ Poisson(W mu_t). By Bayes
    theorem the posterior is Gamma(a + sum x_t, a + sum mu_t), with mean (a + sum x)/(a + sum mu).
    """
    a = 1.0 / max(var_w, MIN_VARIANCE)
    return a + float(np.sum(observed)), a + float(np.sum(mu_observed))


def updated_overload(mu_remaining, capacity: int, shape: float, rate: float):
    """P(X_t > C) for later slots given the Gamma(shape, rate) posterior of W.

    The posterior predictive of a Poisson with a Gamma-distributed rate is negative binomial with
    n = shape and p = rate / (rate + mu_t).
    """
    mu_remaining = np.asarray(mu_remaining, dtype=float)
    return stats.nbinom.sf(capacity, shape, rate / (rate + mu_remaining))


def same_day_update(mu: pd.Series, observed: pd.Series, capacity: int, var_w: float) -> pd.DataFrame:
    """Before/after table for the slots not yet observed today.

    `observed` holds today's counts for the first slots (index = slot labels, in order).
    """
    seen = [p for p in mu.index if p in observed.index]
    rest = [p for p in mu.index if p not in observed.index]
    shape, rate = update_day_factor(mu[seen], observed[seen], var_w)
    before = overload_probability_mixed(mu[rest].to_numpy(), capacity, var_w)
    after = updated_overload(mu[rest].to_numpy(), capacity, shape, rate)
    return pd.DataFrame({"period": rest, "mu": mu[rest].to_numpy(), "mu_updated": mu[rest].to_numpy() * shape / rate,
                         "p_before": before, "p_after": after}).assign(
        w_mean=shape / rate, w_sd=np.sqrt(shape) / rate)


# ===========================================================================
# Extras: dependence checks
# ===========================================================================
def consecutive_slot_correlation(df: pd.DataFrame, model: RateModel, group_col: str = "time_interval") -> pd.DataFrame:
    """Spearman rank correlation of the ratio x / mu between consecutive slots on the same day.

    Dividing by the fitted mean removes the lunch-rush shape and weekday/day-type effects, so the
    correlation measures whether a busy start to the day carries over into later slots (Module 3).
    """
    ratio = (df["arrivals"] / model.fitted.loc[df.index]).rename("ratio")
    wide = df[["date", group_col]].assign(ratio=ratio).pivot(index="date", columns=group_col, values="ratio")
    wide = wide.reindex(columns=sorted(wide.columns))
    rows = []
    for a, b in zip(wide.columns[:-1], wide.columns[1:]):
        pair = wide[[a, b]].dropna()
        if len(pair) >= 3:
            rho, p = stats.spearmanr(pair[a], pair[b])
            rows.append({"pair": f"{a} → {b}", "rho": float(rho), "p_value": float(p), "n": int(len(pair))})
    return pd.DataFrame(rows)


def runs_test(signs) -> dict:
    """Wald-Wolfowitz runs test of randomness for a sequence of +/- signs (Module 5).

    Too few runs means busy days follow busy days (positive day-to-day dependence).
    Normal approximation: R has mean 2 n1 n2 / n + 1 and variance
    2 n1 n2 (2 n1 n2 - n) / (n^2 (n - 1)).
    """
    s = np.asarray(signs, dtype=bool)
    n1, n2 = int(s.sum()), int((~s).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        return {"runs": 1 if n else 0, "expected": float("nan"), "z": float("nan"), "p_value": float("nan"),
                "n_pos": n1, "n_neg": n2}
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mean = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mean) / np.sqrt(var)
    return {"runs": runs, "expected": float(mean), "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z))),
            "n_pos": n1, "n_neg": n2}


def daily_residual_runs(df: pd.DataFrame, model: RateModel) -> dict:
    """Runs test on the sign of (observed - fitted) daily totals, in date order."""
    tmp = df.assign(mu=model.fitted.loc[df.index]).groupby("date")[["arrivals", "mu"]].sum().sort_index()
    return runs_test((tmp["arrivals"] - tmp["mu"]).to_numpy() > 0)


# ===========================================================================
# G11: rolling one-day-ahead backtest
# ===========================================================================
@dataclass
class Backtest:
    slots: pd.DataFrame     # one row per (test day, slot)
    days: pd.DataFrame      # one row per test day
    train_days: int
    update_after: int
    capacity: int


def _peak_hit(predicted: str, actual_counts: pd.Series) -> bool:
    """A predicted peak is a hit if that slot had the day's highest count (ties count as hits)."""
    return bool(actual_counts.get(predicted, -1) == actual_counts.max())


def rolling_backtest(df: pd.DataFrame, capacity: int, train_days: int = DEFAULT_TRAIN_DAYS,
                     update_after: int = 2, group_col: str = "time_interval", seed: int = 0,
                     expanding: bool = True) -> Backtest:
    """For every day after the first `train_days`, fit on earlier days only and predict that day.

    With `expanding=True` the model is fitted on all earlier days (at least `train_days`); otherwise on
    the previous `train_days` days (a rolling window). A day type not yet seen in training (the first
    exam day) is forecast as a regular day, as it would be in practice.

    Two forecasts are compared for each slot:
      * baseline - the method used elsewhere in the project: per-slot mean of the training days,
        plain Poisson tail;
      * model    - regression rate for that weekday and day type, Poisson-Gamma tail (G7 + G10).
    The model forecast is also updated after the first `update_after` slots of the day (G9).
    PIT values are randomised for discrete data: u = F(x - 1) + V (F(x) - F(x - 1)), V ~ U(0, 1).
    """
    if train_days < 5:
        raise ValueError("Use at least 5 training days.")
    dates = sorted(df["date"].unique())
    if len(dates) <= train_days:
        raise ValueError(f"Need more than {train_days} days of data for a backtest (have {len(dates)}).")
    rng = np.random.default_rng(seed)
    types = day_types_of(df)
    work = df.assign(_type=types)
    slot_rows, day_rows = [], []
    for i in range(train_days, len(dates)):
        start = 0 if expanding else i - train_days
        train = work[work["date"].isin(dates[start:i])].drop(columns="_type")
        test = work[work["date"] == dates[i]].sort_values(group_col)
        weekday, day_type = test["day"].iloc[0], test["_type"].iloc[0]
        try:
            model = fit_rate_model(train, group_col)
        except ValueError:
            continue
        type_seen = day_type in model.day_types
        try:
            mu_all = model.predict(weekday, day_type if type_seen else REGULAR)
        except ValueError:
            continue  # weekday not seen in the training window
        var_w = estimate_day_variance(train, model)["var_w"]
        base_all = train.groupby(group_col)["arrivals"].mean()
        test = test[test[group_col].isin(mu_all.index) & test[group_col].isin(base_all.index)]
        if test.empty:
            continue
        periods = test[group_col].tolist()
        x = test["arrivals"].to_numpy(float)
        mu = mu_all.loc[periods].to_numpy()
        base = base_all.loc[periods].to_numpy()
        p_model = overload_probability_mixed(mu, capacity, var_w)
        p_base = stats.poisson.sf(capacity, base)
        lo, hi = predictive_interval(mu, var_w)
        f_hi = predictive_cdf(x, mu, var_w)
        f_lo = predictive_cdf(x - 1, mu, var_w)
        pit = f_lo + rng.random(len(x)) * (f_hi - f_lo)

        p_updated = np.full(len(x), np.nan)
        k = min(update_after, len(x) - 1)
        if k > 0:
            shape, rate = update_day_factor(mu[:k], x[:k], var_w)
            p_updated[k:] = updated_overload(mu[k:], capacity, shape, rate)

        for j, period in enumerate(periods):
            slot_rows.append({"date": dates[i], "day": weekday, "day_type": day_type, "period": period,
                              "actual": x[j], "mu_model": mu[j], "mu_base": base[j],
                              "p_model": p_model[j], "p_base": p_base[j], "p_updated": p_updated[j],
                              "overload": float(x[j] > capacity), "pi_low": lo[j], "pi_high": hi[j],
                              "covered": float(lo[j] <= x[j] <= hi[j]), "pit": pit[j], "after_update": j >= k > 0})
        actual = pd.Series(x, index=periods)
        day_rows.append({"date": dates[i], "day": weekday, "day_type": day_type, "type_seen": type_seen,
                         "var_w": var_w, "n_train_days": int(train["date"].nunique()),
                         "mae_model": float(np.mean(np.abs(x - mu))), "mae_base": float(np.mean(np.abs(x - base))),
                         "peak_model": periods[int(np.argmax(mu))], "peak_base": periods[int(np.argmax(base))],
                         "peak_actual": periods[int(np.argmax(x))],
                         "hit_model": _peak_hit(periods[int(np.argmax(mu))], actual),
                         "hit_base": _peak_hit(periods[int(np.argmax(base))], actual),
                         "n_slots": len(periods), "n_tied_peak": int((x == x.max()).sum())})
    if not day_rows:
        raise ValueError("No test day could be forecast; widen the date filters or shorten the training window.")
    return Backtest(pd.DataFrame(slot_rows), pd.DataFrame(day_rows), train_days, update_after, capacity)


def proportion_z_test(successes: int, n: int, p0: float, alternative: str = "two-sided") -> dict:
    """Large-sample test of a single proportion: z = (p_hat - p0) / sqrt(p0 (1 - p0) / n) (Module 4)."""
    p_hat = successes / n
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    if alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    else:
        p = 2 * stats.norm.sf(abs(z))
    return {"p_hat": float(p_hat), "p0": float(p0), "z": float(z), "p_value": float(p), "n": int(n)}


def brier_score(prob, outcome) -> float:
    """Mean squared difference between forecast probability and the 0/1 outcome (Brier, 1950)."""
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    return float(np.mean((prob - outcome) ** 2))


def reliability_table(prob, outcome, bins: int = 5) -> pd.DataFrame:
    """Forecast probability bins against the observed overload frequency (calibration)."""
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": f"{100 * edges[b]:.0f}–{100 * edges[b + 1]:.0f}%", "mean_forecast": prob[mask].mean(),
                         "observed_rate": outcome[mask].mean(), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def summarize_backtest(bt: Backtest, alpha: float = 0.05, level: float = 0.95) -> dict:
    """All validation statistics and hypothesis tests for the backtest (G11)."""
    s, d = bt.slots, bt.days
    out: dict = {"n_days": int(len(d)), "n_forecasts": int(len(s)), "alpha": alpha}

    # Probability forecasts: Brier score and skill score against the baseline method
    b_model, b_base = brier_score(s["p_model"], s["overload"]), brier_score(s["p_base"], s["overload"])
    out["brier_model"], out["brier_base"] = b_model, b_base
    out["brier_skill"] = 1 - b_model / b_base if b_base > 0 else float("nan")
    upd = s[s["after_update"]]
    if not upd.empty:
        out["brier_updated"] = brier_score(upd["p_updated"], upd["overload"])
        out["brier_not_updated"] = brier_score(upd["p_model"], upd["overload"])

    # Count forecasts: accuracy and correlation (Module 3)
    out["mae_model"], out["mae_base"] = float(d["mae_model"].mean()), float(d["mae_base"].mean())
    r, r_p = stats.pearsonr(s["mu_model"], s["actual"])
    rho, rho_p = stats.spearmanr(s["mu_model"], s["actual"])
    out["pearson_r"], out["pearson_p"], out["spearman_rho"], out["spearman_p"] = float(r), float(r_p), float(rho), float(rho_p)

    groups = np.where(d["day_type"] != REGULAR, d["day_type"] + " days",
                      np.where(d["day"] == "Friday", "Fridays", "Other regular days"))
    out["mae_by_group"] = (d.assign(group=groups).groupby("group")
                           .agg(days=("date", "size"), mae_model=("mae_model", "mean"),
                                mae_base=("mae_base", "mean")).reset_index())

    tests = []
    # Module 4: coverage of the 95% prediction interval
    cov = proportion_z_test(int(s["covered"].sum()), len(s), level)
    out["coverage"] = cov["p_hat"]
    tests.append({"check": f"{int(100 * level)}% prediction intervals cover the actual count", "method":
                  "Large-sample z-test of a single proportion (H₀: coverage = 0.95)", "module": "M4",
                  "statistic": f"z = {cov['z']:.2f}", "p_value": cov["p_value"],
                  "result": f"coverage {100 * cov['p_hat']:.1f}%", "good_if_reject": False})
    # Module 4: peak-slot hit rate against random guessing
    chance = float(np.mean(d["n_tied_peak"] / d["n_slots"]))
    hit = proportion_z_test(int(d["hit_model"].sum()), len(d), chance, "greater")
    out["hit_rate_model"], out["hit_rate_base"], out["hit_chance"] = hit["p_hat"], float(d["hit_base"].mean()), chance
    tests.append({"check": "Predicted peak slot is the actual busiest slot more often than chance",
                  "method": "Large-sample z-test of a single proportion (H₁: hit rate > chance)", "module": "M4",
                  "statistic": f"z = {hit['z']:.2f}", "p_value": hit["p_value"],
                  "result": f"hit rate {100 * hit['p_hat']:.1f}% vs chance {100 * chance:.1f}%",
                  "good_if_reject": True})
    # Module 5: paired comparison of daily errors (model vs baseline)
    diff = d["mae_base"] - d["mae_model"]
    if len(d) >= 3 and diff.std() > 0:
        t_stat, t_p = stats.ttest_rel(d["mae_base"], d["mae_model"], alternative="greater")
    else:
        t_stat, t_p = float("nan"), float("nan")
    tests.append({"check": "Model has a smaller daily error than the per-slot average", "method":
                  "Paired t-test on daily mean absolute error (H₁: baseline error > model error)", "module": "M5",
                  "statistic": f"t = {t_stat:.2f}", "p_value": float(t_p),
                  "result": f"MAE {out['mae_model']:.2f} vs {out['mae_base']:.2f}", "good_if_reject": True})
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if wins + losses else float("nan")
    tests.append({"check": "Model wins on more days than it loses", "method":
                  "Sign test (binomial, H₁: P(model better) > 0.5)", "module": "M5",
                  "statistic": f"{wins} wins, {losses} losses", "p_value": sign_p,
                  "result": f"{wins}/{wins + losses} days better", "good_if_reject": True})
    # Module 5: calibration of the full predictive distribution (PIT should be uniform)
    ks = stats.kstest(s["pit"], "uniform")
    counts, _ = np.histogram(s["pit"], bins=10, range=(0, 1))
    chi = stats.chisquare(counts)
    out["pit_counts"] = counts.tolist()
    tests.append({"check": "Forecast distribution is calibrated (PIT values are uniform)", "method":
                  "Kolmogorov–Smirnov test against U(0, 1)", "module": "M5",
                  "statistic": f"D = {ks.statistic:.3f}", "p_value": float(ks.pvalue),
                  "result": "uniform" if ks.pvalue >= alpha else "not uniform", "good_if_reject": False})
    tests.append({"check": "PIT histogram is flat", "method": "Chi-square goodness of fit, 10 equal bins",
                  "module": "M5", "statistic": f"χ² = {chi.statistic:.2f}, df = 9", "p_value": float(chi.pvalue),
                  "result": "flat" if chi.pvalue >= alpha else "not flat", "good_if_reject": False})
    for t in tests:
        rejected = t["p_value"] < alpha if t["p_value"] == t["p_value"] else None
        t["supports_model"] = None if rejected is None else (rejected == t["good_if_reject"])
    out["tests"] = pd.DataFrame(tests)
    out["reliability_model"] = reliability_table(s["p_model"], s["overload"])
    out["reliability_base"] = reliability_table(s["p_base"], s["overload"])
    return out
