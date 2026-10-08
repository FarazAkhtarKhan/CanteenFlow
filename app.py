"""
app.py — CanteenFlow Streamlit dashboard
=========================================
Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution.

Run with:   streamlit run app.py

This file only handles the user interface. All statistics live in
poisson_model.py, goodness_of_fit.py and analysis.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import visualization as viz
from analysis import DEMAND_METHODS, GRANULARITIES, filter_dataset, run_pipeline
from data_generator import DEFAULT_N_DAYS, DEFAULT_SEED, generate_dataset, save_dataset
from goodness_of_fit import MIN_EXPECTED_FREQUENCY, chi_square_gof, dispersion_test
from poisson_model import overload_breakdown, overload_probability, overload_probability_manual
from utils import (DEFAULT_ALPHA, DEFAULT_DATA_PATH, DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD,
                   clean_dataset, dataset_default_capacity, load_dataset, validate_thresholds)

st.set_page_config(page_title="CanteenFlow — Poisson Canteen Overload Risk", page_icon="🍽️", layout="wide")

RISK_ICON = {"LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM", "HIGH": "🔴 HIGH"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric, button, input { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1400px; }
.hero { padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1rem;
        background: radial-gradient(circle at 10% 20%, rgba(139,124,246,.35), transparent 45%),
                    radial-gradient(circle at 90% 0%, rgba(34,211,238,.25), transparent 40%),
                    linear-gradient(135deg, #151B2E 0%, #0F1426 100%);
        border: 1px solid rgba(139,124,246,.25); box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0;
           background: linear-gradient(90deg, #C4B5FD, #67E8F9); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { color: #CBD5E1; margin: .35rem 0 0 0; font-size: 1.02rem; }
.badge { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
         background: rgba(251,191,36,.12); color: #FCD34D; border: 1px solid rgba(251,191,36,.35); margin-top: .7rem; }
.pipeline { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin: .4rem 0 1rem 0; }
.step { padding: .45rem .8rem; border-radius: 10px; background: #151B2E; border: 1px solid rgba(139,124,246,.3);
        font-size: .85rem; transition: transform .15s ease, border-color .15s ease; }
.step:hover { transform: translateY(-2px); border-color: #22D3EE; }
.arrow { color: #8B7CF6; font-weight: 700; }
.rec { padding: 1rem 1.2rem; border-radius: 14px; margin-bottom: .7rem; background: #151B2E;
       border-left: 4px solid #8B7CF6; transition: transform .15s ease; }
.rec:hover { transform: translateX(3px); }
div[data-testid="stMetric"] { background: #151B2E; border-radius: 14px; padding: .8rem 1rem;
                               transition: transform .15s ease, box-shadow .15s ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(139,124,246,.18); }
</style>
"""


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bundled_dataset() -> pd.DataFrame:
    if not DEFAULT_DATA_PATH.exists():  # first run on a fresh copy: create the dataset
        save_dataset(generate_dataset())
    return load_dataset(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def generate_cached(n_days: int, seed: int) -> pd.DataFrame:
    return generate_dataset(n_days=n_days, seed=seed)


@st.cache_data(show_spinner=False)
def clean_cached(raw: pd.DataFrame):
    return clean_dataset(raw)


@st.cache_data(show_spinner=False)
def pipeline_cached(df, capacity, alpha, low, high, group_col, method):
    return run_pipeline(df, capacity, alpha, low, high, group_col, method)


def pct(p: float) -> str:
    if p != p:  # NaN
        return "n/a"
    return "< 0.01%" if 0 < p < 1e-4 else f"{100 * p:.2f}%"


def how(title: str):
    """Expander used for every 'How is this calculated?' explanation."""
    return st.expander(f"ℹ️ {title}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar_dataset() -> tuple[pd.DataFrame, str]:
    st.sidebar.markdown("## ⚙️ Controls")
    st.sidebar.markdown("### 1 · Dataset")
    source = st.sidebar.radio("Data source", ["Canteen arrival records (12:30 PM – 2:30 PM)",
                                              "Simulate custom period records",
                                              "Upload CSV"], label_visibility="collapsed", key="sidebar_data_source")
    if source == "Simulate custom period records":
        n_days = st.sidebar.number_input("Working days to simulate", 30, 365, DEFAULT_N_DAYS, step=5, key="sidebar_n_days")
        seed = st.sidebar.number_input("Random seed", 0, 10_000_000, DEFAULT_SEED, step=1, key="sidebar_seed")
        return generate_cached(int(n_days), int(seed)), f"Simulated records ({n_days} days, seed {seed})"
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CSV with columns date, time_interval, arrivals "
                                            "(optional: day, hour, period, capacity)", type="csv", key="sidebar_upload")
        if uploaded is None:
            st.info("⬅️ Upload a CSV file in the sidebar to continue.")
            st.stop()
        return load_dataset(uploaded), f"Uploaded file: {uploaded.name}"
    return load_bundled_dataset(), "Canteen arrival records (12:30 PM – 2:30 PM)"


def sidebar_parameters(clean_df: pd.DataFrame) -> dict:
    st.sidebar.markdown("### 2 · Model parameters")
    capacity = st.sidebar.number_input("Capacity C (arrivals per 15-min interval)", min_value=1, max_value=1000,
                                       value=dataset_default_capacity(clean_df), step=1,
                                       help="Arrivals the canteen can reasonably handle in one 15-minute interval.",
                                       key="sidebar_capacity")
    alpha = st.sidebar.select_slider("Significance level α", options=[0.01, 0.025, 0.05, 0.10], value=DEFAULT_ALPHA,
                                     key="sidebar_alpha")
    low = st.sidebar.slider("LOW-risk threshold: P(X > C) below (%)", 1, 98, int(100 * DEFAULT_LOW_THRESHOLD),
                            key="sidebar_low_thresh")
    high = st.sidebar.slider("HIGH-risk threshold: P(X > C) above (%)", 2, 99, int(100 * DEFAULT_HIGH_THRESHOLD),
                             key="sidebar_high_thresh")
    st.sidebar.caption("Thresholds are project-defined decision rules, not universal scientific standards.")
    try:
        low_p, high_p = validate_thresholds(low / 100, high / 100)
    except ValueError as err:
        st.sidebar.error(str(err))
        st.error("Please make the LOW-risk threshold smaller than the HIGH-risk threshold.")
        st.stop()

    st.sidebar.markdown("### 3 · Analysis options")
    granularity = st.sidebar.radio("Time-period granularity", list(GRANULARITIES), horizontal=True,
                                   help="15-minute intervals are the primary unit. 'Hourly' pools the four "
                                        "15-minute intervals of each hour.", key="sidebar_granularity")
    method = st.sidebar.selectbox("Demand-level method", list(DEMAND_METHODS), format_func=DEMAND_METHODS.get,
                                  key="sidebar_demand_method")

    st.sidebar.markdown("### 4 · Filters")
    min_d, max_d = clean_df["date"].min().date(), clean_df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d,
                                       key="sidebar_date_range")
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Select both a start and an end date in the sidebar.")
        st.stop()
    all_days = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                if d in set(clean_df["day"])]
    days = st.sidebar.multiselect("Weekdays", all_days, default=all_days, key="sidebar_weekdays")
    return {"capacity": int(capacity), "alpha": float(alpha), "low": low_p, "high": high_p,
            "group_col": GRANULARITIES[granularity], "granularity": granularity, "method": method,
            "start": date_range[0], "end": date_range[1], "days": days}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def header(source_label: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="hero"><h1>🍽️ CanteenFlow</h1>
        <p><b>Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution</b></p>
        <p>Operating Hours: <b>12:30 PM – 2:30 PM (Lunch Service)</b> | 15-Minute Interval Arrival & Overload Analysis</p>
        <p>A statistical probability-estimation system: λ is estimated from arrival
        counts and used to compute P(arrivals &gt; capacity) for every time period.</p>
        </div>""", unsafe_allow_html=True)


def section_overview(result, df, report, params) -> None:
    k = result.kpis
    cols = st.columns(5)
    cols[0].metric("Total observations", f"{k['total_observations']:,}", border=True,
                   help=f"{k['n_days']} days × {k['n_periods']} periods, after cleaning.")
    cols[1].metric("Average arrivals / interval", f"{k['avg_arrivals']:.2f}", border=True)
    cols[2].metric("Max observed arrivals", f"{k['max_arrivals']}", border=True, help=k["max_arrivals_when"])
    cols[3].metric("Highest-demand period", k["peak_period"], f"λ̂ = {k['peak_lambda']:.2f}", delta_color="off",
                   border=True)
    cols[4].metric("Highest overload probability", pct(k["max_overload_prob"]), k["max_overload_period"],
                   delta_color="off", border=True)

    st.markdown("#### Research pipeline")
    steps = ["Data Collection & Ingestion", "Data Cleaning", "Exploratory Analysis", "Estimate λ",
             "Mean vs Variance", "Poisson Goodness-of-Fit", "P(X > Capacity)", "Risk Classification",
             "Visualization", "Recommendations"]
    st.markdown('<div class="pipeline">' + '<span class="arrow">→</span>'.join(
        f'<span class="step">{i + 1}. {s}</span>' for i, s in enumerate(steps)) + "</div>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(viz.overload_by_period(result.table, params["low"], params["high"]), width="stretch",
                        key="overview_overload_chart")
    with right:
        st.plotly_chart(viz.risk_distribution(result.table), width="stretch", key="overview_risk_chart")

    with how("Data cleaning report (pipeline step 2)"):
        st.markdown(
            f"- Rows read: **{report['rows_in']}**  \n"
            f"- Missing arrival counts dropped: **{report['missing_arrivals']}** (recording gaps)  \n"
            f"- Invalid counts (negative / non-integer) dropped: **{report['invalid_arrivals']}**  \n"
            f"- Unreadable dates / intervals dropped: **{report['invalid_dates']} / {report['invalid_intervals']}**  \n"
            f"- Duplicate (date, interval) records dropped: **{report['duplicates']}**  \n"
            f"- Rows used: **{report['rows_out']}** (after date/weekday filters: **{len(df)}**)\n\n"
            "Missing counts are **dropped, not imputed**: filling them with the mean would artificially "
            "reduce the variance and bias the mean-vs-variance check.")


def section_arrivals(result, df, params, selected) -> None:
    table, group_col = result.table, params["group_col"]
    with how("Why Poisson?"):
        st.markdown(
            "The variable being modelled is the **count of arrivals in a fixed time interval** (15 minutes). "
            "The Poisson distribution is the standard model for such counts when arrivals occur independently "
            "and at an approximately constant average rate within the interval. Its single parameter λ is both "
            "the mean and the variance. These assumptions are *checked* below rather than taken for granted.")
        st.latex(r"P(X = x) = \frac{e^{-\lambda}\,\lambda^{x}}{x!},\qquad x = 0, 1, 2, \dots")

    st.plotly_chart(viz.lambda_by_period(table, params["capacity"]), width="stretch", key="arrivals_lambda_chart")
    with how("How is λ calculated?"):
        st.latex(r"\hat{\lambda} = \frac{\sum_{i=1}^{n} x_i}{n} = \frac{\text{total arrivals}}{\text{number of observations}}")
        row = table.set_index("period").loc[selected]
        st.markdown(f"For **{selected}**: λ̂ = {row['total_arrivals']} / {row['n']} = **{row['lambda_hat']:.3f}**. "
                    f"This is the maximum-likelihood estimate of λ. The error bars are exact (Garwood) "
                    f"{100 * (1 - params['alpha']):.0f}% confidence intervals, using the fact that a sum of "
                    "Poisson counts is itself Poisson:")
        st.latex(r"\left[\frac{\chi^2_{\alpha/2}(2S)}{2n},\ \frac{\chi^2_{1-\alpha/2}(2S+2)}{2n}\right],\quad S=\sum x_i")
    with how("How are Low / Medium / High demand periods determined?"):
        st.markdown(f"Method: **{DEMAND_METHODS[result.demand_info['method']]}**.  \n"
                    f"Rule applied to this data: {result.demand_info['rule']}  \n"
                    "Tertiles split the periods into three equal-sized groups by λ̂ (a relative ranking). "
                    "The z-score method marks periods more than half a standard deviation below/above the "
                    "average λ̂. The highest-demand period is simply the one with the largest λ̂. "
                    "Demand level describes arrival *rate*; overload *risk* (which also depends on capacity) "
                    "is computed separately.")

    observations = df.loc[df[group_col] == selected, "arrivals"].to_numpy()
    lam = float(table.set_index("period").loc[selected, "lambda_hat"])
    st.plotly_chart(viz.arrival_distribution(observations, lam, f"Arrival count distribution — {selected} "
                                             f"(n = {observations.size})"), width="stretch", key="arrivals_dist_chart")
    st.caption("Bars: how often each count was observed. Line: the frequency expected under the fitted "
               "Poisson(λ̂) model (n × P(X = x)). Change the period in the sidebar.")

    st.markdown("#### Mean vs variance")
    tab_scatter, tab_bars = st.tabs(["Scatter (Var = Mean line)", "Side-by-side bars"])
    with tab_scatter:
        if table["variance"].notna().any():
            st.plotly_chart(viz.mean_vs_variance_scatter(table), width="stretch", key="arrivals_mv_scatter_chart")
        else:
            st.info("At least two observations per period are needed to compute a variance.")
    with tab_bars:
        st.plotly_chart(viz.mean_vs_variance_bars(table), width="stretch", key="arrivals_mv_bars_chart")
    with how("What does Mean ≈ Variance show (and not show)?"):
        st.markdown(
            "For a Poisson distribution **E(X) = Var(X) = λ**, so the dispersion index Var/Mean should be close "
            "to 1. Values > 1 suggest *overdispersion*, values < 1 *underdispersion*. Because sample values always "
            "fluctuate, each period is checked with the **index-of-dispersion test**:")
        st.latex(r"D = \frac{(n-1)\,s^2}{\bar{x}} \sim \chi^2_{n-1}\ \text{under } H_0\text{ (two-sided test)}")
        st.markdown("Mean ≈ variance is a *property* of the Poisson distribution, **not proof** that data are "
                    "Poisson — other distributions can share it. The chi-square goodness-of-fit test (next tab) "
                    "gives stronger evidence about the whole shape of the distribution.")

    st.plotly_chart(viz.arrivals_heatmap(df), width="stretch", key="arrivals_heatmap_chart")

    st.markdown("#### Descriptive statistics by period")
    st.dataframe(
        table[["period", "n", "total_arrivals", "mean", "variance", "std_dev", "lambda_hat", "lambda_ci_low",
               "lambda_ci_high", "var_mean_ratio", "mean_var_ratio", "dispersion_p", "dispersion", "demand_level"]],
        hide_index=True, width="stretch",
        column_config={
            "period": "Period", "n": "n", "total_arrivals": "Σx",
            "mean": st.column_config.NumberColumn("Mean", format="%.3f"),
            "variance": st.column_config.NumberColumn("Variance (s²)", format="%.3f"),
            "std_dev": st.column_config.NumberColumn("Std dev", format="%.3f"),
            "lambda_hat": st.column_config.NumberColumn("λ̂", format="%.3f"),
            "lambda_ci_low": st.column_config.NumberColumn("λ CI low", format="%.2f"),
            "lambda_ci_high": st.column_config.NumberColumn("λ CI high", format="%.2f"),
            "var_mean_ratio": st.column_config.NumberColumn("Var/Mean", format="%.3f"),
            "mean_var_ratio": st.column_config.NumberColumn("Mean/Var", format="%.3f"),
            "dispersion_p": st.column_config.NumberColumn("Dispersion p", format="%.4f"),
            "dispersion": "Dispersion test", "demand_level": "Demand level"},
        key="arrivals_summary_dataframe")


def section_gof(result, df, params, selected) -> None:
    alpha, group_col = params["alpha"], params["group_col"]
    if group_col == "hour_block":
        st.warning("**Hourly view:** pooling four 15-minute intervals assumes the arrival rate is constant within "
                   "the hour. Where the rate changes inside the hour (e.g. the lunch build-up), the pooled counts "
                   "are a *mixture* of Poisson distributions, which is overdispersed — so the test may reject H0 "
                   "even if every 15-minute interval is Poisson. Use the 15-minute view for model checking.")

    st.markdown(f"#### Chi-square goodness-of-fit — {selected}")
    st.markdown("**H₀:** the arrival counts follow a Poisson distribution.  \n"
                "**H₁:** the arrival counts do not follow a Poisson distribution.")
    observations = df.loc[df[group_col] == selected, "arrivals"].to_numpy()
    gof = chi_square_gof(observations, alpha)
    disp = dispersion_test(observations, alpha)

    cols = st.columns(5)
    cols[0].metric("λ̂", f"{gof.lambda_hat:.3f}", border=True)
    cols[1].metric("χ² statistic", f"{gof.chi_square:.3f}" if gof.valid else "n/a", border=True)
    cols[2].metric("Degrees of freedom", gof.df if gof.valid else "n/a", border=True)
    cols[3].metric("p-value", f"{gof.p_value:.4f}" if gof.valid else "n/a", border=True)
    cols[4].metric("Decision (α = %g)" % alpha, gof.decision, border=True)
    (st.error if gof.reject_h0 else st.warning if not gof.valid else st.success)(gof.message)

    if not gof.table.empty:
        left, right = st.columns([1, 1.2])
        with left:
            show = gof.table[["category", "observed", "expected"] +
                             (["contribution"] if "contribution" in gof.table else [])]
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "category": "Category (x)", "observed": "Observed O",
                "expected": st.column_config.NumberColumn("Expected E = n·P", format="%.2f"),
                "contribution": st.column_config.NumberColumn("(O−E)²/E", format="%.3f")},
                key="gof_categories_dataframe")
            for note in gof.merge_notes:
                st.caption(note)
        with right:
            st.plotly_chart(viz.gof_observed_vs_expected(gof.table), width="stretch", key="gof_obs_vs_exp_chart")

    st.markdown(f"Supporting evidence — index-of-dispersion test: Var/Mean = **{disp['index']:.3f}**, "
                f"p = **{disp['p_value']:.4f}** → *{disp['label']}*.")

    with how("Why Chi-square?"):
        st.markdown("To statistically evaluate whether the **observed frequency distribution** of counts is "
                    "reasonably consistent with the **fitted Poisson distribution**. It compares the whole shape, "
                    "not only the mean and variance.")
    with how("How is χ² calculated, and how are categories combined?"):
        st.latex(r"\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i},\qquad E_i = n \cdot P(X \in \text{category } i \mid \hat\lambda)")
        st.latex(r"df = k - 1 - 1 \quad(\text{one extra df lost because } \lambda \text{ is estimated})")
        st.markdown(
            f"1. Start from single count values 0, 1, 2, … covering *every* possible count.\n"
            f"2. Merge adjacent values from left to right until each category has **expected frequency "
            f"≥ {MIN_EXPECTED_FREQUENCY:g}** (Cochran's rule — the χ² approximation is unreliable for small E).\n"
            "3. A leftover right-tail category with E < 5 is merged into the previous one.\n"
            "4. So the first category is 'X ≤ a' and the last 'X ≥ b', and the expected probabilities sum to 1.\n"
            "5. If fewer than 3 categories remain (df < 1), the test is reported as not valid.\n\n"
            "Binning depends only on λ̂ and n — never on the observed counts. The p-value is P(χ²_df ≥ observed χ²). "
            "If p > α we **fail to reject** H₀ (the data are consistent with Poisson — this is not proof); "
            "if p ≤ α we reject H₀.")

    st.markdown("#### Goodness of fit for all periods")
    g = result.gof_summary
    st.markdown(
        f"H₀ rejected in **{g['rejections']} of {g['tested']}** valid tests. Even if every period were truly "
        f"Poisson, about **α × m = {g['expected_by_chance']:.1f}** rejections are expected by chance alone "
        f"(multiple testing). Probability of ≥ {g['rejections']} rejections under H₀ everywhere "
        f"(binomial): **{g['binom_p']:.3f}** → " +
        ("**more rejections than chance explains — the Poisson assumption is questionable.**" if g["excessive"]
         else "**consistent with chance — no overall evidence against the Poisson model.**"))
    st.dataframe(result.gof_table, hide_index=True, width="stretch", column_config={
        "period": "Period", "lambda_hat": st.column_config.NumberColumn("λ̂", format="%.3f"),
        "categories": "Categories k", "chi_square": st.column_config.NumberColumn("χ²", format="%.3f"),
        "df": "df", "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
        "decision": "Decision", "valid": "Valid test"},
        key="gof_all_periods_dataframe")


def section_overload(result, params, selected) -> None:
    capacity = params["capacity"]
    row = result.table.set_index("period").loc[selected]
    st.markdown(f"#### Overload prediction — {selected}")
    cols = st.columns(4)
    cols[0].metric("Capacity C", capacity, border=True)
    cols[1].metric("λ̂", f"{row['lambda_hat']:.3f}", border=True)
    cols[2].metric("P(X > C)", pct(row["overload_prob"]), border=True,
                   help=f"Range over the λ confidence interval: {pct(row['overload_prob_low'])} – "
                        f"{pct(row['overload_prob_high'])}")
    cols[3].metric("Risk level", RISK_ICON[row["risk"]], border=True)
    st.caption(f"Uncertainty: evaluating P(X > C) at the ends of the λ confidence interval gives "
               f"{pct(row['overload_prob_low'])} to {pct(row['overload_prob_high'])}.")

    with how("How is overload probability calculated?"):
        st.latex(r"P(X > C) = 1 - P(X \le C) = 1 - \sum_{x=0}^{C} \frac{e^{-\hat\lambda}\,\hat\lambda^{x}}{x!}")
        lam = float(row["lambda_hat"])
        breakdown = overload_breakdown(lam, capacity)
        st.dataframe(breakdown, hide_index=True, height=240, width="stretch", column_config={
            "P(X = x)": st.column_config.NumberColumn(format="%.6f"),
            "P(X ≤ x)": st.column_config.NumberColumn(format="%.6f")},
            key="overload_breakdown_dataframe")
        manual = overload_probability_manual(lam, capacity)
        st.markdown(f"P(X ≤ {capacity}) = {breakdown['P(X ≤ x)'].iloc[-1]:.6f}  →  "
                    f"P(X > {capacity}) = 1 − {breakdown['P(X ≤ x)'].iloc[-1]:.6f} = **{manual:.6f}** (manual formula).  \n"
                    f"SciPy cross-check `poisson.sf({capacity}, {lam:.3f})` = **{overload_probability(lam, capacity):.6f}**.")

    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(viz.overload_by_period(result.table, params["low"], params["high"]), width="stretch",
                        key="overload_by_period_chart")
    with right:
        st.plotly_chart(viz.overload_curve(float(row["lambda_hat"]), capacity, selected), width="stretch",
                        key="overload_curve_chart")

    st.markdown("#### Risk table")
    risk_table = result.table[["period", "lambda_hat", "capacity", "overload_prob", "risk"]].copy()
    risk_table["overload_pct"] = 100 * risk_table["overload_prob"]
    risk_table["risk"] = risk_table["risk"].map(RISK_ICON)
    st.dataframe(risk_table[["period", "lambda_hat", "capacity", "overload_pct", "risk"]], hide_index=True,
                 width="stretch", column_config={
                     "period": "Period", "lambda_hat": st.column_config.NumberColumn("λ̂", format="%.2f"),
                     "capacity": "Capacity",
                     "overload_pct": st.column_config.ProgressColumn("P(X > Capacity)", format="%.2f%%",
                                                                     min_value=0, max_value=100),
                     "risk": "Risk"},
                 key="overload_risk_dataframe")
    with how("How are risk levels decided?"):
        st.markdown(f"- **LOW**: P(X > C) < {100 * params['low']:.0f}%\n"
                    f"- **MEDIUM**: {100 * params['low']:.0f}% ≤ P(X > C) ≤ {100 * params['high']:.0f}%\n"
                    f"- **HIGH**: P(X > C) > {100 * params['high']:.0f}%\n\n"
                    "These are **project-defined decision thresholds**, adjustable in the sidebar. They are not "
                    "universal scientific standards; a canteen manager could choose stricter or looser values.")


def section_recommendations(result) -> None:
    st.markdown("#### Operational recommendations (model-based)")
    for rec in result.recommendations:
        st.markdown(f'<div class="rec">\n\n{rec}\n\n</div>', unsafe_allow_html=True)


def section_method(df, result) -> None:
    st.markdown("#### Academic transparency")
    st.markdown(
        "| Component | What it is in this project |\n|---|---|\n"
        "| **Observed data** | Arrival counts across 15-minute intervals (12:30 PM – 2:30 PM). |\n"
        "| **Statistical model** | Poisson distribution for the number of arrivals in a 15-minute interval. |\n"
        "| **Estimation** | λ̂ = Σx / n for each period, with an exact confidence interval. |\n"
        "| **Model checking** | Mean-vs-variance (dispersion test) and chi-square goodness-of-fit. |\n"
        "| **Prediction** | P(X > C) = 1 − P(X ≤ C) under the fitted model — a probability, not an exact forecast. |\n"
        "| **Decision** | LOW / MEDIUM / HIGH using configurable, project-defined thresholds. |")
    st.markdown("This is a **statistical probability estimation system**, not an AI/ML system; it does not predict "
                "exactly how many students will arrive.")
    st.markdown("#### Limitations")
    st.markdown(
        "1. Arrivals may vary due to unexpected timetable changes or special events.\n"
        "2. Real student arrivals may not follow a perfect Poisson process.\n"
        "3. Poisson assumes a stable rate within each interval.\n"
        "4. Independence may not strictly hold when students arrive in groups.\n"
        "5. Over- or under-dispersion can occur.\n"
        "6. Exams, holidays, weather, and campus events affect arrivals.\n"
        "7. P(X > C) depends on the estimated λ and the chosen capacity.\n"
        "8. Risk thresholds are project-defined, not universal standards.")
    st.markdown("#### Data")
    preview = df.drop(columns=["hour_block"]).assign(date=df["date"].dt.date)
    st.dataframe(preview, hide_index=True, width="stretch", height=300, key="method_data_preview_dataframe")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download filtered dataset (CSV)", preview.to_csv(index=False), "canteen_arrivals_filtered.csv",
                       "text/csv", key="btn_download_filtered_csv")
    c2.download_button("⬇️ Download period results (CSV)", result.table.to_csv(index=False), "canteenflow_results.csv",
                       "text/csv", key="btn_download_results_csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        raw, source_label = sidebar_dataset()
        clean_df, report = clean_cached(raw)
    except ValueError as err:  # includes EmptyDatasetError and missing columns
        st.markdown(CSS, unsafe_allow_html=True)
        st.error(f"Could not load the dataset: {err}")
        st.stop()

    params = sidebar_parameters(clean_df)
    header(source_label)

    df = filter_dataset(clean_df, params["start"], params["end"], params["days"])
    if df.empty:
        st.warning("No observations match the selected date range / weekdays. Widen the filters in the sidebar.")
        st.stop()

    try:
        result = pipeline_cached(df, params["capacity"], params["alpha"], params["low"], params["high"],
                                 params["group_col"], params["method"])
    except ValueError as err:
        st.error(f"Analysis could not be completed: {err}")
        st.stop()

    periods = result.table["period"].tolist()
    st.sidebar.markdown("### 5 · Period for detailed view")
    selected = st.sidebar.selectbox("Time period", periods, index=periods.index(result.kpis["peak_period"]),
                                    help="Used for the histogram, goodness-of-fit and worked overload calculation.",
                                    key="sidebar_selected_period")

    tabs = st.tabs(["📊 1 · Overview", "📈 2 · Arrival Analysis", "🧪 3 · Poisson Goodness of Fit",
                    "⚠️ 4 · Overload Prediction", "💡 5 · Recommendations", "🗂️ Data & Method"])
    with tabs[0]:
        section_overview(result, df, report, params)
    with tabs[1]:
        section_arrivals(result, df, params, selected)
    with tabs[2]:
        section_gof(result, df, params, selected)
    with tabs[3]:
        section_overload(result, params, selected)
    with tabs[4]:
        section_recommendations(result)
    with tabs[5]:
        section_method(df, result)


main()
