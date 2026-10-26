"""
make_figures.py
---------------
Builds every figure and number used in the IEEE paper directly from the project
code, so the paper always matches the dashboard.

    python paper/make_figures.py        # writes paper/figures/*.pdf and paper/results.json

Requires matplotlib in addition to the project dependencies (see paper/requirements.txt).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forecast as fc  # noqa: E402
from analysis import run_pipeline  # noqa: E402
from goodness_of_fit import chi_square_gof  # noqa: E402
from poisson_model import required_capacity  # noqa: E402
from utils import clean_dataset, load_dataset  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
CAPACITY, ALPHA = 30, 0.05

# Project palette (navy, teal, risk colours); print-safe on white
INK, TEAL, SOFT, GREY = "#1B2A4A", "#00868B", "#9FD4D6", "#8A94A3"
RISK = {"LOW": "#2E8B57", "MEDIUM": "#E07B2E", "HIGH": "#C0392B"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": "#5B6576", "xtick.color": "#2A3442", "ytick.color": "#2A3442",
    "grid.color": "#DDE3EA", "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = 3.45  # IEEE column width, inches


def short(period: str) -> str:
    return period[:5]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, report = clean_dataset(load_dataset())
    res = run_pipeline(df, CAPACITY, ALPHA)
    table = res.table.sort_values("period").reset_index(drop=True)
    periods = table["period"].tolist()
    x = np.arange(len(periods))
    numbers: dict = {"report": report, "kpis": {k: v for k, v in res.kpis.items() if k != "risk_counts"},
                     "risk_counts": res.kpis["risk_counts"], "gof": res.gof_summary,
                     "quiet": {k: v for k, v in res.quiet.items() if k != "ranking"}}

    # ---- Fig. 2: arrival rate per slot with exact CI -------------------------------------------
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    err = np.vstack([table["lambda_hat"] - table["lambda_ci_low"], table["lambda_ci_high"] - table["lambda_hat"]])
    ax.bar(x, table["lambda_hat"], width=0.62, color=TEAL, zorder=2)
    ax.errorbar(x, table["lambda_hat"], yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    ax.axhline(CAPACITY, color=RISK["HIGH"], lw=0.9, ls="--", zorder=1)
    ax.text(len(x) - 0.5, CAPACITY + 0.8, f"capacity $C$ = {CAPACITY}", ha="right", color=RISK["HIGH"], fontsize=7)
    ax.set_xticks(x, [short(p) for p in periods])
    ax.set_ylabel(r"$\hat\lambda_t$ (arrivals / 15 min)")
    ax.set_xlabel("Slot start time")
    ax.set_ylim(0, 40)
    ax.grid(axis="y", zorder=0)
    save(fig, "rates")

    # ---- Fig. 3: goodness of fit for the peak slot --------------------------------------------
    peak = res.kpis["peak_period"]
    obs = df.loc[df["time_interval"] == peak, "arrivals"].to_numpy()
    gof = chi_square_gof(obs, ALPHA)
    k = np.arange(obs.min() - 2, obs.max() + 3)
    counts = np.array([(obs == v).sum() for v in k])
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.bar(k, counts, width=0.8, color=SOFT, label="Observed days", zorder=2)
    ax.plot(k, len(obs) * stats.poisson.pmf(k, gof.lambda_hat), color=INK, lw=1.2, marker="o", ms=2.2,
            label=rf"Poisson($\hat\lambda$ = {gof.lambda_hat:.2f})", zorder=3)
    ax.set_xlabel(f"Arrivals in slot {peak.replace('-', '–')}")
    ax.set_ylabel("Number of days")
    ax.legend(loc="upper right")
    ax.grid(axis="y", zorder=0)
    save(fig, "gof_peak")
    numbers["gof_peak"] = {"period": peak, "chi2": gof.chi_square, "df": gof.df, "p": gof.p_value,
                           "lambda": gof.lambda_hat, "n": int(len(obs))}

    # ---- Forecasting model ----------------------------------------------------------------------
    model = fc.fit_rate_model(df)
    disp = fc.estimate_day_variance(df, model)
    effects = fc.day_effect_tests(df)
    v = disp["var_w"]
    numbers["model"] = {"r2": model.r_squared, "adj_r2": model.adj_r_squared, "n": model.n, "dof": model.dof,
                        "smear": model.smear, "sigma": model.sigma,
                        "coef": model.coefficients.to_dict(orient="records")}
    numbers["dispersion"], numbers["effects"] = disp, effects

    # ---- Fig. 4: overload probability, Poisson vs Poisson-Gamma ----------------------------------
    p_pois = table["overload_prob"].to_numpy()
    p_mix = fc.overload_probability_mixed(table["lambda_hat"].to_numpy(), CAPACITY, v)
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    w = 0.38
    ax.bar(x - w / 2, 100 * p_pois, width=w, color=TEAL, label="Poisson", zorder=2)
    ax.bar(x + w / 2, 100 * p_mix, width=w, color="white", edgecolor=INK, lw=0.8, hatch="////",
           label="Poisson–Gamma", zorder=2)
    for yv in (20, 50):
        ax.axhline(yv, color=GREY, lw=0.7, ls=":", zorder=1)
    ax.text(len(x) - 0.45, 51.5, "high-risk threshold (50%)", ha="right", fontsize=6.5, color=GREY)
    ax.text(len(x) - 0.45, 21.5, "low-risk threshold (20%)", ha="right", fontsize=6.5, color=GREY)
    ax.set_xticks(x, [short(p) for p in periods])
    ax.set_ylabel(r"$P(X_t > C)$, %")
    ax.set_xlabel("Slot start time")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    ax.grid(axis="y", zorder=0)
    save(fig, "overload")
    numbers["overload"] = {p: {"poisson": float(a), "mixed": float(b)} for p, a, b in zip(periods, p_pois, p_mix)}
    numbers["required_capacity_20"] = {p: required_capacity(float(l), 0.20)
                                       for p, l in zip(periods, table["lambda_hat"])}

    # ---- Fig. 5: forecast for three kinds of day + peak probability ------------------------------
    cases = [("Monday", "Regular", INK, "-", "o"), ("Friday", "Regular", TEAL, "--", "s"),
             ("Monday", "Exam", RISK["MEDIUM"], ":", "^")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL * 2.02, 1.85))
    numbers["forecast"] = {}
    for wd, dt, color, ls, mk in cases:
        mu = model.predict(wd, dt)
        po = fc.overload_probability_mixed(mu.to_numpy(), CAPACITY, v)
        pk = fc.peak_time_distribution(mu, CAPACITY, v)
        a1.plot(x, 100 * po, color=color, ls=ls, marker=mk, ms=3, lw=1.1, label=f"{wd}, {dt.lower()} week")
        numbers["forecast"][f"{wd}-{dt}"] = {
            "mu": mu.round(3).to_dict(), "p_over": dict(zip(periods, map(float, po))),
            "p_peak": pk["p_peak"].round(4).to_dict(), "most_likely": pk["most_likely_peak"],
            "p_most_likely": pk["p_most_likely"], "top_two": pk["top_two"], "p_top_two": pk["p_top_two"],
            "p_any": pk["p_any_overload"], "expected_overloaded": pk["expected_overloaded_slots"],
            "peak_q": pk["peak_count_quantiles"]}
    a1.set_xticks(x, [short(p) for p in periods], rotation=45)
    a1.set_ylabel(r"Predicted $P(X_t > C)$, %")
    a1.set_ylim(0, 100)
    a1.set_title("(a) Day-specific overload forecast", loc="left")
    a1.legend(loc="upper right")
    a1.grid(axis="y")
    pk = numbers["forecast"]["Monday-Regular"]["p_peak"]
    vals = np.array([pk[p] for p in periods])
    a2.bar(x, 100 * vals, width=0.62, color=[INK if val == vals.max() else SOFT for val in vals], zorder=2)
    for xi, val in zip(x, vals):
        if val >= 0.05:
            a2.text(xi, 100 * val + 1.2, f"{100 * val:.0f}%", ha="center", fontsize=6.5, color=INK)
    a2.set_xticks(x, [short(p) for p in periods], rotation=45)
    a2.set_ylabel("P(slot is the day's peak), %")
    a2.set_ylim(0, 50)
    a2.set_title("(b) Peak-time probability, regular Monday", loc="left")
    a2.grid(axis="y", zorder=0)
    fig.tight_layout(w_pad=1.5)
    save(fig, "forecast")

    # ---- Fig. 6: same-day Bayesian update --------------------------------------------------------
    mu = model.predict("Monday")
    observed = pd.Series({p: int(round(mu[p] * 1.15)) for p in mu.index[:2]})
    upd = fc.same_day_update(mu, observed, CAPACITY, v)
    xu = np.arange(len(upd))
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.bar(xu - 0.2, 100 * upd["p_before"], width=0.38, color=SOFT, label="Forecast at start of day", zorder=2)
    ax.bar(xu + 0.2, 100 * upd["p_after"], width=0.38, color=INK, label="Updated after two slots", zorder=2)
    ax.set_xticks(xu, [short(p) for p in upd["period"]])
    ax.set_ylabel(r"$P(X_t > C)$, %")
    ax.set_xlabel("Remaining slot start time")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", zorder=0)
    save(fig, "update")
    numbers["update"] = {"observed": observed.to_dict(), "table": upd.round(4).to_dict(orient="records")}

    # ---- Fig. 7: validation on unseen days -------------------------------------------------------
    bt = fc.rolling_backtest(df, CAPACITY, train_days=30, update_after=2)
    s = fc.summarize_backtest(bt, ALPHA)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL * 2.02, 1.9))
    a1.plot([0, 100], [0, 100], color=GREY, ls=":", lw=0.8, label="Perfect calibration")
    for rel, name, color, mk in ((s["reliability_base"], "Per-slot average", TEAL, "o"),
                                 (s["reliability_model"], "Regression model", INK, "D")):
        a1.plot(100 * rel["mean_forecast"], 100 * rel["observed_rate"], color=color, marker=mk, ms=3.5, lw=1.1,
                label=name)
    a1.set_xlabel(r"Forecast $P(X_t > C)$, %")
    a1.set_ylabel("Observed overload frequency, %")
    a1.set_xlim(0, 100)
    a1.set_ylim(0, 100)
    a1.set_title("(a) Reliability diagram", loc="left")
    a1.legend(loc="upper left")
    a1.grid()
    edges = np.linspace(0, 1, 11)
    a2.bar((edges[:-1] + edges[1:]) / 2, s["pit_counts"], width=0.09, color=SOFT, zorder=2)
    a2.axhline(np.mean(s["pit_counts"]), color=INK, ls="--", lw=0.9, label="Uniform (calibrated)")
    a2.set_xlabel("PIT value")
    a2.set_ylabel("Number of slot forecasts")
    a2.set_title("(b) PIT histogram", loc="left")
    a2.legend(loc="upper left")
    a2.grid(axis="y", zorder=0)
    fig.tight_layout(w_pad=1.5)
    save(fig, "validation")
    tests = s["tests"].drop(columns=["good_if_reject"])
    numbers["backtest"] = {k: s[k] for k in ("n_days", "n_forecasts", "brier_model", "brier_base", "brier_skill",
                                             "brier_updated", "brier_not_updated", "mae_model", "mae_base",
                                             "pearson_r", "pearson_p", "spearman_rho", "coverage",
                                             "hit_rate_model", "hit_rate_base", "hit_chance", "pit_counts")}
    numbers["backtest"]["tests"] = tests.to_dict(orient="records")
    numbers["backtest"]["mae_by_group"] = s["mae_by_group"].to_dict(orient="records")
    numbers["dependence"] = {"consecutive": fc.consecutive_slot_correlation(df, model).to_dict(orient="records"),
                             "runs": fc.daily_residual_runs(df, model)}
    numbers["table"] = table[["period", "n", "lambda_hat", "lambda_ci_low", "lambda_ci_high", "variance",
                              "overload_prob", "risk", "gof_p_value"]].to_dict(orient="records")

    (Path(__file__).resolve().parent / "results.json").write_text(json.dumps(numbers, indent=1, default=float))
    write_tables(numbers, p_mix)
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}, results.json and generated tables")


def _p(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_tables(numbers: dict, p_mix) -> None:
    """LaTeX table bodies, so no number in the paper is typed by hand."""
    gen = Path(__file__).resolve().parent / "generated"
    gen.mkdir(exist_ok=True)
    risk = {"LOW": "Low", "MEDIUM": "Med.", "HIGH": "High"}
    rows = []
    for r, pm in zip(numbers["table"], p_mix):
        rows.append(f"{r['period'][:5]}--{r['period'][6:]} & {r['lambda_hat']:.2f} & "
                    f"[{r['lambda_ci_low']:.2f}, {r['lambda_ci_high']:.2f}] & {r['variance'] / r['lambda_hat']:.2f} & "
                    f"{r['gof_p_value']:.3f} & {100 * r['overload_prob']:.2f} & {100 * pm:.2f} & {risk[r['risk']]} \\\\")
    (gen / "table_slots.tex").write_text("\n".join(rows) + "\n")

    names = {"Intercept": "Intercept$^{\\dagger}$"}
    rows = []
    for c in numbers["model"]["coef"]:
        term = names.get(c["term"], c["term"][:10].replace("slot ", "Slot ") if c["term"].startswith("slot")
                         else c["term"])
        neg = lambda v, fmt: format(v, fmt).replace("-", "$-$")  # noqa: E731  proper minus sign
        rows.append(f"{term} & {neg(c['estimate'], '.4f')} & {c['std_error']:.4f} & {neg(c['t'], '.2f')} & "
                    f"{_p(c['p_value'])} & "
                    f"{c['multiplier']:.3f} \\\\")
    (gen / "table_coef.tex").write_text("\n".join(rows) + "\n")

    label = {
        "95% prediction intervals cover the actual count": ("95\\% interval coverage", "$z$-test, one proportion"),
        "Predicted peak slot is the actual busiest slot more often than chance": ("Peak slot named correctly",
                                                                                "$z$-test, one proportion"),
        "Model has a smaller daily error than the per-slot average": ("Lower daily MAE than baseline", "Paired $t$-test"),
        "Model wins on more days than it loses": ("More winning days than baseline", "Sign test"),
        "Forecast distribution is calibrated (PIT values are uniform)": ("PIT uniform", "Kolmogorov--Smirnov"),
        "PIT histogram is flat": ("PIT histogram flat", "$\\chi^2$ goodness of fit"),
    }
    rows = []
    for t in numbers["backtest"]["tests"]:
        name, method = label[t["check"]]
        stat = (t["statistic"].replace("χ²", "$\\chi^2$").replace("z = ", "$z$ = ").replace("t = ", "$t$ = ")
                .replace("D = ", "$D$ = ").replace("df", "df").replace("-", "$-$"))
        result = t["result"].replace("%", "\\%")
        rows.append(f"{name} & {method} & {stat} & {_p(t['p_value'])} & {result} & "
                    f"{'Yes' if t['supports_model'] else 'No'} \\\\")
    (gen / "table_tests.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
