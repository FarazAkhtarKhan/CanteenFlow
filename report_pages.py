"""
report_pages.py
---------------
One function per dashboard page. app.py registers them with st.navigation and
fills ui.CTX before the selected page runs.

Project      Overview, Literature Review, Research Gaps, Syllabus Map, Methodology
Analysis     Data & EDA, Arrival Rate, Goodness of Fit, Overload Probability, Simulation Check
Forecast     Peak-Time Forecast, Forecast Validation
Results      Risk & Recommendations, Least-Crowded Time
Conclusion   Limitations & Future Scope, References & Downloads
"""

from __future__ import annotations

import html
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import forecast as fc
import ui
import visualization as viz
from analysis import DEMAND_METHODS, comfort_probabilities, quietest_in_window, weekday_rates
from goodness_of_fit import MIN_EXPECTED_FREQUENCY, chi_square_gof, dispersion_test
from literature import GAPS, METHOD_REFERENCES, PAPERS, SYLLABUS_MAP, SYLLABUS_MODULES, paper
from poisson_model import (classify_risk, overload_breakdown, overload_probability, overload_probability_manual,
                           required_capacity, simulate_overload_frequency)
from ui import (chart, ctx, figures, html_block, note, page_header, pct, pill,
                select_period_from_chart, slot, two_columns, verdict)

TEAM = [("Mayur Krishna", "26MCA0191"), ("Faraz Akthar Khan", "26MCA0197"), ("Malhar Bhosle", "26MCA0199")]


def _row(period: str) -> pd.Series:
    return ctx().result.table.set_index("period").loc[period]


def _link(page_key: str, label: str) -> None:
    page = ctx().pages.get(page_key)
    if page is not None:
        st.page_link(page, label=label)


# ===========================================================================
# PROJECT
# ===========================================================================
def overview() -> None:
    c = ctx()
    k, q, p = c.result.kpis, c.result.quiet, c.params
    team = "".join(f"<b>{name}</b> ({reg})<br>" for name, reg in TEAM)
    html_block(
        '<div class="cover"><h1>Statistical Estimation of Canteen Arrival Overload Probability Using Poisson '
        'Distribution</h1><p class="sub">CanteenFlow: an interactive statistical analysis of lunch-time arrivals</p>'
        '<div class="meta">'
        '<div><span>Course</span><b>PAMAT501, Probability and Statistics</b></div>'
        f'<div><span>Presented by</span>{team}</div>'
        '<div><span>Faculty guide</span><b>Dr. Manimaran</b></div>'
        '<div><span>Base paper</span><b>Tang et al. (2019), Physica A, 521, 29–40</b></div>'
        '</div></div>')

    st.markdown("#### Abstract")
    st.markdown(
        f"University canteens see sharp changes in arrival intensity during lunch. This project estimates, for "
        f"each {c.minutes}-minute slot, the probability that student arrivals exceed the counters' capacity. "
        f"Using {k['n_days']} working days of arrival counts ({k['total_observations']:,} slot observations), the "
        f"arrival rate λₜ is estimated for every slot, the Poisson assumption is tested with dispersion and "
        f"chi-square goodness-of-fit tests, and P(Xₜ > C) is computed for a capacity of C = {p['capacity']}. "
        f"The riskiest slot is **{slot(k['max_overload_period'])}** (P = {pct(k['max_overload_prob'], 1)}); the "
        f"least crowded is **{slot(q['best'])}** (λ̂ = {q['best_lambda']:.1f}). The second review extends this "
        "to **prediction**: a regression gives day-specific rates, simulation gives the probability that each "
        "slot is the day's peak, Bayes theorem updates the forecast from the morning's counts, and a rolling "
        "backtest tests the forecasts on unseen days. Every method maps to the PAMAT501 syllabus.")
    if c.synthetic:
        st.caption("The bundled records are synthetic prototype data with a realistic lunch-rush shape. Field "
                   "counts can be loaded from the sidebar (Dataset, Upload a CSV) once collected.")

    figures([(f"{k['total_observations']:,}", "slot observations"), (f"{k['n_days']}", "working days"),
             (f"{k['n_periods']}", f"time slots of {c.minutes} min"),
             (f"{k['peak_lambda']:.1f}", f"highest λ̂ ({slot(k['peak_period'])})"),
             (pct(k['max_overload_prob'], 1), "highest P(overload)"),
             (slot(q['best']), "least-crowded slot")])

    st.markdown("#### Overload risk by time slot")
    left, right = two_columns()
    with left:
        key = f"ov_chart_{c.period}"
        chart(viz.overload_by_period(c.result.table, p["low"], p["high"]), key,
              on_select=partial(select_period_from_chart, key), selection_mode="points")
    with right:
        row = _row(c.period)
        html_block(f'<div class="panel"><div class="title">{slot(c.period)}</div>'
                   f'<h4>Expected arrivals λ̂</h4><p>{row["lambda_hat"]:.2f} '
                   f'(CI {row["lambda_ci_low"]:.1f} to {row["lambda_ci_high"]:.1f})</p>'
                   f'<h4>P(X > {p["capacity"]})</h4><p>{pct(row["overload_prob"])} {pill(row["risk"])}</p>'
                   f'<h4>Poisson fit</h4><p>{row["gof_decision"]} (p = {row["gof_p_value"]:.3f})</p></div>')
        note("Interact", "Click any bar to load that slot here and on every analysis page. You can also pick a "
                         "slot under <i>Slot to inspect</i> in the sidebar.")

    st.markdown("#### Project direction")
    html_block('<div class="flow"><span>Observed arrivals</span><i>→</i><span>Estimate λₜ</span><i>→</i>'
               '<span>Test Poisson fit</span><i>→</i><span>P(Xₜ > C)</span><i>→</i><span>Risk label</span>'
               '<i>→</i><span>Least-crowded time</span><i>→</i><span>Predict the peak</span><i>→</i>'
               '<span>Validate on new days</span></div>')
    st.markdown("#### Objectives")
    objectives = [
        ("Collect and organise arrival data", "Arrival counts at fixed 5-, 10- or 15-minute intervals across m days, "
                                              "validated and cleaned.", "data"),
        ("Estimate time-specific rates", "λ̂ₜ with exact confidence intervals, and a test of Poisson suitability.",
         "rate"),
        ("Calculate overload probability", "P(Xₜ > C) for a chosen capacity, with its uncertainty.", "overload"),
        ("Identify useful time windows", "Low / medium / high risk labels and the least-crowded slot.", "quietest"),
        ("Predict the peak time", "Day-specific overload probabilities, P(slot is the peak) and same-day "
                                  "updating (second review).", "forecast"),
        ("Validate the forecasts", "One-day-ahead backtest with syllabus hypothesis tests (second review).",
         "validation"),
    ]
    cols = st.columns(2, gap="medium")
    for i, (title, text, page_key) in enumerate(objectives):
        with cols[i % 2], st.container(border=True):
            st.markdown(f"**{i + 1}. {title}**  \n{text}")
            _link(page_key, "Open the analysis")


def literature() -> None:
    page_header("Project", "Literature Review",
                f"{len(PAPERS)} studies reviewed: the base paper, nine more from the first review (canteen "
                "modelling, queueing, arrival-process statistics, time-varying demand, campus crowd sensing) and "
                "eight added for the second review on forecasting arrivals and checking forecasts. "
                "<b>Select a row</b> to read a study's focus, method, findings and what it leaves open.")
    categories = sorted({p["category"] for p in PAPERS})
    chosen = st.pills("Filter by area", categories, selection_mode="multi", default=categories, key="lit_filter")
    shown = [p for p in PAPERS if p["category"] in (chosen or categories)]
    table = pd.DataFrame([{"Study": p["cite"] + (" (base paper)" if p.get("base") else ""), "Year": p["year"],
                           "Area": p["category"], "Main focus": p["focus"]} for p in shown])
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key=f"lit_table_{len(shown)}", column_config={"Year": st.column_config.NumberColumn(format="%d")})
    rows = event.selection.rows if event and event.selection else []
    selected = shown[rows[0]] if rows else (shown[0] if shown else None)
    if selected:
        doi = (f'<a href="https://doi.org/{selected["doi"]}" target="_blank">doi.org/{selected["doi"]}</a>'
               if selected["doi"] else "No DOI listed")
        html_block(
            f'<div class="panel"><div class="title">{html.escape(selected["cite"])}</div>'
            f'<p>{html.escape(selected["reference"])} {doi}</p><div class="grid2">'
            f'<div><h4>Focus</h4><p>{html.escape(selected["focus"])}</p></div>'
            f'<div><h4>Method</h4><p>{html.escape(selected["method"])}</p></div>'
            f'<div><h4>Main finding</h4><p>{html.escape(selected["finding"])}</p></div>'
            f'<div><h4>Relevance to this project</h4><p>{html.escape(selected["relevance"])}</p></div></div>'
            f'<h4>What it leaves open for this project</h4><p>{html.escape(selected["leaves_open"])}</p></div>')

    st.markdown("#### Positioning of this work")
    st.markdown(
        "| Group of studies | What they model | What this project adds |\n|---|---|---|\n"
        "| Canteen pedestrian flow (Tang et al., 2019a, 2019b) | Full meal process: movement, windows, seating | "
        "A compact arrival-risk indicator per time slot |\n"
        "| Canteen queueing (Wang, 2011; Yuan et al., 2023) | Windows and capacity with assumed Poisson arrivals | "
        "A per-slot test of whether arrivals are Poisson |\n"
        "| Arrival-process statistics (Brown et al., 2005; Kim & Whitt, 2014) | Testing and forecasting Poisson "
        "arrivals in call centres and hospitals | The same discipline applied to a campus canteen |\n"
        "| Time-varying demand (Green et al., 2007; Leemis, 1991) | Rates that change within the day | Short-slot "
        "rates with exact confidence intervals |\n"
        "| Model checking (Cameron & Trivedi, 1990) | Overdispersion tests for Poisson counts | "
        "Index-of-dispersion test for every slot |\n"
        "| Campus crowd sensing (Variskhanov & Arreeras, 2026) | Wi-Fi-based day-level crowd badges | "
        "Slot-level probabilities from simple arrival counts |\n"
        "| Campus dining forecasting (Shah et al., 2025) | Daily customer totals with machine learning | "
        "Slot-level forecasts of overload and peak time with syllabus statistics |\n"
        "| Arrival forecasting (Weinberg et al., 2007; Shen & Huang, 2008; Aldor-Noiman et al., 2009; "
        "Matteson et al., 2011; Ibrahim et al., 2016) | One-day-ahead and same-day forecasts of Poisson rates "
        "for staffing | Peak-time probability, Bayesian same-day update and a backtest for a canteen |\n"
        "| Forecast verification (McNulty, 1988) | Brier score and skill against a reference | "
        "Scoring P(X > C) against the per-slot average |")
    _link("gaps", "Continue to the research gaps")


def gaps() -> None:
    page_header("Project", "Research Gaps",
                "Eleven gaps identified from the survey. G1 was stated at the first review; G2 to G6 come from "
                "the extended review; G7 to G11 come from the second-review reading on forecasting. Each is a "
                "<b>focus and application</b> gap, not a claim that Poisson "
                "modelling of arrivals is new. Pick a gap to see what it means, the evidence for it, and where "
                "this project addresses it.")
    labels = [f'{g["id"]} {g["title"].split()[0].rstrip("-")}' for g in GAPS]
    if st.session_state.get("gap_pick") not in labels:
        st.session_state["gap_pick"] = labels[0]
    choice = st.segmented_control("Research gap", labels, key="gap_pick", required=True,
                                  label_visibility="collapsed") or st.session_state["gap_pick"]
    g = GAPS[labels.index(choice)]
    evidence = "".join(f'<li>{html.escape(paper(k)["cite"])}: {html.escape(paper(k)["leaves_open"])}</li>'
                       for k in g["evidence"])
    html_block(
        f'<div class="panel"><div class="title">{g["id"]}. {html.escape(g["title"])}</div>'
        f'<p><b>{html.escape(g["summary"])}</b></p>'
        f'<h4>What the gap is</h4><p>{html.escape(g["explain"])}</p>'
        f'<h4>Evidence from the literature</h4><ul>{evidence}</ul>'
        f'<h4>How this project addresses it</h4><p>{html.escape(g["addressed"])}</p></div>')
    _link(g["page"], "See it in the analysis")

    st.markdown("#### All gaps at a glance")
    st.markdown("| Gap | In one line | Addressed by |\n|---|---|---|\n" + "\n".join(
        f'| **{x["id"]}** {x["title"]} | {x["summary"]} | {x["addressed"]} |' for x in GAPS))


def methodology() -> None:
    c = ctx()
    page_header("Project", "Methodology",
                "The statistical framework in nine steps (seven from the first review, two added for forecasting), "
                "the formulas behind each test, and an interactive "
                "Poisson explorer. The framework follows the first-review plan; the Poisson model is "
                "<b>checked against the data, not assumed by default</b>.")
    steps = [
        ("Observe", "Count arrivals at the canteen entry (manual count or permitted video) in fixed slots across m days.",
         "data_generator.generate_dataset (prototype data), utils.load_dataset (field CSV)", "Raw arrival records"),
        ("Clean", "Drop missing, negative, fractional, unreadable or duplicate counts. Missing counts are dropped, "
                  "not imputed, so the variance is not artificially shrunk.", "utils.clean_dataset",
         "Clean records and a cleaning report"),
        ("Group", "Separate records by time slot (or pool into hours to show the effect of coarse intervals).",
         "analysis.summarize_by_group", "Counts per slot"),
        ("Estimate", "λ̂ₜ = Σx / m per slot, with an exact Garwood confidence interval.",
         "poisson_model.estimate_lambda, poisson_model.lambda_confidence_interval", "λ̂ₜ with CI"),
        ("Test fit", "Index-of-dispersion test (Var = Mean) and chi-square goodness of fit with categories "
                     "pooled to E ≥ 5; binomial check for multiple testing.",
         "goodness_of_fit.dispersion_test, goodness_of_fit.chi_square_gof", "Fit verdict per slot"),
        ("Compute", "P(Xₜ > C) = 1 − P(Xₜ ≤ C), computed by the textbook sum and by SciPy, cross-checked by "
                    "tests and Monte Carlo simulation.",
         "poisson_model.overload_probability, poisson_model.simulate_overload_frequency", "Overload probability"),
        ("Present", "Risk labels, least-crowded slot with an exact comparison test, recommendations, charts.",
         "poisson_model.classify_risk, analysis.quietest_slots, analysis.generate_recommendations",
         "This dashboard"),
        ("Forecast", "Regression of log counts on slot, weekday and day type gives λ for a chosen day; Poisson–Gamma "
                     "probabilities, simulated P(peak), and a Bayesian same-day update (gaps G7–G10).",
         "forecast.fit_rate_model, forecast.estimate_day_variance, forecast.peak_time_distribution, "
         "forecast.same_day_update", "Predicted overload risk and peak time"),
        ("Validate", "Forecast each day from earlier days only and test the forecasts: Brier skill, coverage and "
                     "hit-rate z-tests, paired t and sign tests, KS and chi-square tests of the PIT (gap G11).",
         "forecast.rolling_backtest, forecast.summarize_backtest", "Evidence that the forecasts work"),
    ]
    labels = [f"{i + 1} {s[0]}" for i, s in enumerate(steps)]
    if st.session_state.get("method_step") not in labels:
        st.session_state["method_step"] = labels[3]
    choice = st.segmented_control("Framework step", labels, key="method_step", required=True,
                                  label_visibility="collapsed") or st.session_state["method_step"]
    name, what, code, output = steps[labels.index(choice)]
    code_html = " ".join(f"<code>{part}</code>" for part in code.split(", "))
    html_block(f'<div class="panel"><div class="title">Step {choice}</div><p>{what}</p><div class="grid2">'
               f'<div><h4>Implemented in</h4><p>{code_html}</p></div>'
               f'<div><h4>Output</h4><p>{output}</p></div></div></div>')

    st.markdown("#### Mathematical foundation")
    tabs = st.tabs(["Poisson model", "Estimating λ", "Overload probability", "Dispersion test",
                    "Goodness of fit", "Comparing two slots", "Regression forecast", "Day-to-day variation",
                    "Same-day update", "Scoring forecasts"])
    with tabs[0]:
        st.markdown("Xₜ is the number of students arriving during slot t. The model assumes")
        st.latex(r"X_t \sim \mathrm{Poisson}(\lambda_t),\qquad P(X_t = x) = \frac{e^{-\lambda_t}\lambda_t^{x}}{x!},"
                 r"\quad x = 0, 1, 2, \dots")
        st.latex(r"E[X_t] = \mathrm{Var}(X_t) = \lambda_t")
        st.markdown("λₜ is estimated separately for each slot because arrival intensity changes across the peak.")
    with tabs[1]:
        st.markdown("For slot t observed on m days, the maximum-likelihood estimate is the sample mean:")
        st.latex(r"\ln L(\lambda) = -m\lambda + \ln\lambda\sum x_i - \sum \ln x_i!\ \Rightarrow\ "
                 r"\hat\lambda_t = \frac{1}{m}\sum_{i=1}^{m} x_i")
        st.markdown("Exact (Garwood, 1936) confidence interval, using S = Σxᵢ ~ Poisson(mλ):")
        st.latex(r"\left[\frac{\chi^2_{\alpha/2}(2S)}{2m},\ \frac{\chi^2_{1-\alpha/2}(2S+2)}{2m}\right]")
    with tabs[2]:
        st.markdown("Overload means more than C students arrive in a slot. With K = C + 1 this is the "
                    "first-review notation P(Xₜ ≥ K):")
        st.latex(r"P(X_t > C) = P(X_t \ge K) = 1 - \sum_{x=0}^{C} \frac{e^{-\hat\lambda_t}\hat\lambda_t^{x}}{x!}")
        st.markdown(f"Risk label: Low if P < {100 * c.params['low']:.0f}%, High if P > {100 * c.params['high']:.0f}%, "
                    "Medium otherwise (thresholds set in the sidebar).")
    with tabs[3]:
        st.markdown("Index-of-dispersion test of Var = Mean, two-sided:")
        st.latex(r"D = \frac{(m-1)s^2}{\bar x} \sim \chi^2_{m-1}\ \text{under } H_0")
        st.markdown("Var/Mean > 1 indicates overdispersion (for example group arrivals); < 1 underdispersion.")
    with tabs[4]:
        st.markdown("H₀: counts follow Poisson(λ̂). H₁: they do not.")
        st.latex(r"\chi^2 = \sum_{i=1}^{k}\frac{(O_i - E_i)^2}{E_i},\qquad df = k - 2")
        st.markdown(f"Adjacent counts are merged until every expected frequency is at least "
                    f"{MIN_EXPECTED_FREQUENCY:g} (Cochran, 1954). One df is lost because probabilities sum to 1 and "
                    "one because λ is estimated.")
    with tabs[5]:
        st.markdown("To check that the least-crowded slot is genuinely quieter than the runner-up "
                    "(Przyborowski & Wileński, 1940), condition on the combined total:")
        st.latex(r"S_1 \mid (S_1 + S_2) \sim \mathrm{Binomial}\!\left(S_1 + S_2,\ \frac{m_1}{m_1 + m_2}\right)"
                 r"\ \text{under } H_0: \lambda_1 = \lambda_2")
        st.markdown("The one-sided p-value is P(Binomial ≤ S₁).")
    with tabs[6]:
        st.markdown("Multiple linear regression (Module 3) on the log of the counts, with dummy variables for "
                    "slot t, weekday d and day type e (reference levels dropped):")
        st.latex(r"\ln(x_{dt} + 0.5) = \beta_0 + \alpha_t + \gamma_{d} + \delta_{e} + \varepsilon_{dt},\qquad "
                 r"\hat\beta = (X^\top X)^{-1}X^\top y")
        st.latex(r"\hat\lambda_{dt} = \exp(x_{dt}^\top\hat\beta)\cdot\overline{e^{\hat\varepsilon}} - 0.5,\qquad "
                 r"t_j = \hat\beta_j / \mathrm{SE}(\hat\beta_j) \sim t_{n-p}")
        st.markdown("Weekday effects are also tested on daily totals with one-way ANOVA and Kruskal-Wallis, exam "
                    "days with Welch's t-test and Mann-Whitney U (Module 5).")
    with tabs[7]:
        st.markdown("Each day has a busyness factor W with mean 1 and variance v. By the theorem of total "
                    "probability (Module 1) with W ~ Gamma (Module 2):")
        st.latex(r"X \mid W \sim \mathrm{Poisson}(W\lambda),\ W \sim \mathrm{Gamma}(1/v, 1/v)\ \Rightarrow\ "
                 r"E[X] = \lambda,\ \mathrm{Var}(X) = \lambda + v\lambda^2")
        st.latex(r"\hat v = \frac{\chi^2_P - (n - p)}{\sum \hat\lambda},\qquad "
                 r"\chi^2_P = \sum \frac{(x - \hat\lambda)^2}{\hat\lambda} \sim \chi^2_{n-p}\ \text{if Poisson holds}")
    with tabs[8]:
        st.markdown("Bayes theorem (Module 1). Prior W ~ Gamma(a, a), a = 1/v. After today's first k slots:")
        st.latex(r"W \mid x_1,\dots,x_k \sim \mathrm{Gamma}\Big(a + \sum_{t\le k} x_t,\ a + \sum_{t\le k}\lambda_t\Big)")
        st.markdown("Each later slot then has a negative binomial predictive distribution, which gives the updated "
                    "P(Xₜ > C). P(slot t is the peak) is estimated by simulating whole days.")
    with tabs[9]:
        st.markdown("Overload in a slot is a yes/no event o ∈ {0, 1} with forecast probability p (Brier, 1950; "
                    "McNulty, 1988):")
        st.latex(r"B = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2,\qquad \mathrm{BSS} = 1 - \frac{B_{\text{model}}}"
                 r"{B_{\text{per-slot average}}}")
        st.latex(r"z = \frac{\hat p - p_0}{\sqrt{p_0(1 - p_0)/n}}\quad\text{(coverage, hit rate; Module 4)}")

    _poisson_explorer(float(_row(c.period)["lambda_hat"]), c.params["capacity"], c.period)


@st.fragment
def _poisson_explorer(lam_slot: float, capacity: int, period: str) -> None:
    st.markdown("#### Interactive Poisson explorer")
    lam_max = float(max(60.0, round(lam_slot * 1.6)))
    cap_max = int(max(80, capacity * 2))
    if "pe_lam" not in st.session_state:
        st.session_state["pe_lam"] = round(lam_slot, 1)
        st.session_state["pe_cap"] = capacity

    def use_slot():
        st.session_state["pe_lam"] = round(lam_slot, 1)
        st.session_state["pe_cap"] = capacity

    c1, c2, c3 = st.columns([1, 1, 0.7], vertical_alignment="bottom")
    lam = c1.slider("Arrival rate λ", 0.5, lam_max, step=0.1, key="pe_lam")
    cap = c2.slider("Capacity C", 1, cap_max, step=1, key="pe_cap")
    c3.button(f"Load slot {slot(period)}", on_click=use_slot, width="stretch")
    p_over = float(stats.poisson.sf(cap, lam))
    left, right = two_columns()
    with left:
        figures([(pct(p_over), f"P(X > {cap})"), (pct(1 - p_over), f"P(X ≤ {cap})"),
                 (f"{lam:.1f}", "mean = variance"), (f"{np.sqrt(lam):.2f}", "standard deviation √λ")])
        chart(viz.poisson_explorer(lam, cap), "explorer_chart")
    with right:
        note("How to read it", "Each bar is P(X = x). Red bars are outcomes above capacity; their total is the "
                               "overload probability.")
        note("Try this", "Raise λ: the distribution shifts right and spreads by √λ. Raise C near the mean: the red "
                         "tail shrinks quickly, so a small capacity increase at the peak buys a large drop in risk.")


# ===========================================================================
# ANALYSIS
# ===========================================================================
def data_eda() -> None:
    c = ctx()
    r, group_col = c.report, c.params["group_col"]
    page_header("Analysis", "Data and Exploratory Analysis",
                f"The arrival records behind every result: where they come from, how they were cleaned, and their "
                f"shape across {c.minutes}-minute slots and weekdays. Source: <b>{html.escape(c.raw_label)}</b>.")
    figures([(f"{r['rows_in']}", "records read"), (f"{r['missing_arrivals']}", "missing counts dropped"),
             (f"{r['invalid_arrivals'] + r['invalid_dates'] + r['invalid_intervals']}", "invalid rows dropped"),
             (f"{r['duplicates']}", "duplicates dropped"), (f"{len(c.df)}", "observations analysed")])
    with st.expander("Data collection protocol and file format"):
        st.markdown(
            "- Arrivals are counted at the canteen entry point (manual count or permitted video), following the "
            "observation style of Tang et al. (2019a).\n"
            "- Fixed 5-, 10- or 15-minute slots during the lunch peak, recorded on m separate working days.\n"
            "- Attributes: date, day, time slot, arrival count (optional: day type such as Regular / Exam, "
            "period, capacity).\n"
            "- CSV columns: `date, time_interval, arrivals` (optional `day_type`), e.g. "
            "`2026-01-05, 12:30-12:40, 14, Regular`.\n"
            "- Missing counts are dropped, not filled in: imputing the mean would shrink the variance and bias the "
            "dispersion test.")
        if c.synthetic:
            st.info("Field data collection is in progress. Until it is complete, the analysis uses synthetic "
                    "prototype records generated with a seeded random generator; they are not observed results.")

    tabs = st.tabs(["Distribution by slot", "Weekday profile", "Day × slot heatmap", "3D view", "Records"])
    with tabs[0]:
        left, right = two_columns()
        with left:
            chart(viz.slot_boxplot(c.df, group_col), "eda_box")
        with right:
            note("How to read it", "Each box holds the middle 50% of daily counts for a slot; the line is the "
                                   "median and the dashed line the mean. Taller boxes mean more day-to-day "
                                   "variation. The rise and fall across slots is the lunch rush.")
            note("Why it matters", "Arrival intensity clearly changes within the lunch period, so a single λ for "
                                   "the whole period would be wrong. That is why λ is estimated per slot.")
    with tabs[1]:
        left, right = two_columns()
        with left:
            chart(viz.weekday_profile(c.df, group_col), "eda_weekday")
        with right:
            note("How to read it", "One line per weekday. Click a day in the legend to hide it; double-click to "
                                   "show that day alone. Lines that run close together mean the weekday has "
                                   "little effect.")
    with tabs[2]:
        left, right = two_columns()
        with left:
            chart(viz.arrivals_heatmap(c.df), "eda_heatmap")
        with right:
            note("How to read it", "Rows are days, columns slots; darker cells mean more arrivals. A dark "
                                   "vertical band is a slot that is busy on most days. Blank cells are dropped "
                                   "recording gaps.")
    with tabs[3]:
        left, right = two_columns()
        with left:
            chart(viz.arrivals_surface_3d(c.df), "eda_surface")
        with right:
            note("How to read it", "Every recorded count as a landscape: slots run left to right, days front to "
                                   "back, height and colour are arrivals. The ridge is the lunch rush; dips across "
                                   "a whole day are quiet days such as exam weeks. Drag to rotate.")
    with tabs[4]:
        preview = c.df.drop(columns=["hour_block"]).assign(date=c.df["date"].dt.date)
        st.dataframe(preview, hide_index=True, width="stretch", height=420)
        st.caption(f"{len(preview)} records after cleaning and the sidebar filters.")


def arrival_rate() -> None:
    c = ctx()
    table, row = c.result.table, _row(c.period)
    page_header("Analysis", "Arrival Rate per Slot (λ̂ₜ)",
                f"The expected number of arrivals in each {c.minutes}-minute slot, with exact confidence intervals. "
                "Bars above the capacity line are slots where an average day already exceeds what the counters can "
                "serve. <b>Click a bar</b> to inspect that slot.")
    if c.params["group_col"] == "hour_block":
        st.info("Hourly view: λ̂ is the mean count per slot within each hour. Pooling slots hides the rise and fall "
                "inside the hour (research gap G3); switch back to individual slots in the sidebar.")
    left, right = two_columns()
    with left:
        key = f"rate_chart_{c.period}"
        chart(viz.lambda_by_period(table, c.params["capacity"], c.period), key,
              on_select=partial(select_period_from_chart, key), selection_mode="points")
    with right:
        note("How to read it", "Bar height is λ̂; the whisker is its confidence interval, the range of rates "
                               "consistent with the data. The outlined bar is the selected slot.")
        note("Demand level", f"Colour ranks slots by λ̂ ({html.escape(DEMAND_METHODS[c.result.demand_info['method']])}). "
                             f"{html.escape(c.result.demand_info['rule'])}")

    st.markdown(f"#### Worked estimate for {slot(c.period)}")
    st.latex(rf"\hat\lambda = \frac{{\sum x_i}}{{m}} = \frac{{{int(row['total_arrivals'])}}}{{{int(row['n'])}}}"
             rf" = {row['lambda_hat']:.3f}")
    st.latex(rf"{100 * (1 - c.params['alpha']):.0f}\%\ \text{{CI}} = [{row['lambda_ci_low']:.2f},\ "
             rf"{row['lambda_ci_high']:.2f}]")
    st.markdown("#### Summary statistics by slot")
    st.dataframe(
        table[["period", "n", "total_arrivals", "mean", "variance", "std_dev", "lambda_hat", "lambda_ci_low",
               "lambda_ci_high", "min", "max", "demand_level"]],
        hide_index=True, width="stretch",
        column_config={"period": "Slot", "n": "Days m", "total_arrivals": "Σx",
                       "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                       "variance": st.column_config.NumberColumn("Variance s²", format="%.2f"),
                       "std_dev": st.column_config.NumberColumn("SD", format="%.2f"),
                       "lambda_hat": st.column_config.NumberColumn("λ̂", format="%.3f"),
                       "lambda_ci_low": st.column_config.NumberColumn("CI low", format="%.2f"),
                       "lambda_ci_high": st.column_config.NumberColumn("CI high", format="%.2f"),
                       "min": "Min", "max": "Max", "demand_level": "Demand level"})


def goodness_of_fit() -> None:
    c = ctx()
    alpha, group_col, table = c.params["alpha"], c.params["group_col"], c.result.table
    page_header("Analysis", "Goodness of Fit",
                "Every probability in this project assumes Poisson counts, so the assumption is tested for each "
                "slot (research gap G2): a chi-square goodness-of-fit test on the whole distribution and an "
                "index-of-dispersion test on mean versus variance.")
    if group_col == "hour_block":
        st.warning("Hourly slots pool several short slots whose rates differ, which produces overdispersion even "
                   "when every short slot is Poisson (Kim & Whitt, 2014). Use individual slots for model checking.")
    observations = c.df.loc[c.df[group_col] == c.period, "arrivals"].to_numpy()
    gof = chi_square_gof(observations, alpha)
    disp = dispersion_test(observations, alpha)

    tabs = st.tabs([f"Chi-square test: {slot(c.period)}", "Mean vs variance", "All slots"])
    with tabs[0]:
        st.markdown("**H₀:** arrival counts follow Poisson(λ̂).  **H₁:** they do not.")
        left, right = two_columns()
        with left:
            chart(viz.arrival_distribution(observations, gof.lambda_hat, ""), "fit_hist")
        with right:
            figures([(f"{gof.chi_square:.2f}" if gof.valid else "n/a", "χ² statistic"),
                     (f"{gof.df}" if gof.valid else "n/a", "degrees of freedom"),
                     (f"{gof.p_value:.3f}" if gof.valid else "n/a", "p-value")], stacked=True)
            verdict("na" if not gof.valid else ("bad" if gof.reject_h0 else "ok"), html.escape(gof.message))
        note("How to read it", "Bars count the days on which each number of arrivals was seen; the line is what "
                               "the fitted Poisson model expects. Change the slot in the sidebar to test another.")
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            if not gof.table.empty:
                cols = ["category", "observed", "expected"] + (["contribution"] if "contribution" in gof.table else [])
                st.dataframe(gof.table[cols], hide_index=True, width="stretch", column_config={
                    "category": "Category (x)", "observed": "Observed O",
                    "expected": st.column_config.NumberColumn("Expected E", format="%.2f"),
                    "contribution": st.column_config.NumberColumn("(O−E)²/E", format="%.3f")})
            for line in gof.merge_notes:
                st.caption(line)
        with c2:
            if not gof.table.empty:
                chart(viz.gof_observed_vs_expected(gof.table), "fit_categories")
    with tabs[1]:
        left, right = two_columns()
        with left:
            if table["variance"].notna().any():
                chart(viz.mean_vs_variance_scatter(table), "fit_scatter")
        with right:
            note("How to read it", "Each point is a slot. Poisson counts have variance equal to the mean, so "
                                   "points should lie near the dashed line. Green: no significant difference.")
            note(f"{slot(c.period)}: Var/Mean = {disp['index']:.2f}",
                 f"D = {disp['statistic']:.1f} on {disp['df']} df, two-sided p = {disp['p_value']:.3f}: "
                 f"{disp['label']}. Equal mean and variance is necessary for a Poisson model but not proof of one, "
                 "which is why the chi-square test is also run.")
    with tabs[2]:
        g = c.result.gof_summary
        figures([(f"{g['rejections']} of {g['tested']}", "slots where H₀ was rejected"),
                 (f"{g['expected_by_chance']:.1f}", "rejections expected by chance (α × m)"),
                 (f"{g['binom_p']:.3f}", "P(at least this many | all Poisson)")])
        verdict("bad" if g["excessive"] else "ok",
                "More rejections than chance explains: the Poisson assumption is questionable."
                if g["excessive"] else "Rejections are in line with chance: no overall evidence against the "
                                       "Poisson model.")
        st.dataframe(c.result.gof_table, hide_index=True, width="stretch", column_config={
            "period": "Slot", "n": "Days m", "lambda_hat": st.column_config.NumberColumn("λ̂", format="%.3f"),
            "categories": "Categories k", "chi_square": st.column_config.NumberColumn("χ²", format="%.3f"),
            "df": "df", "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
            "decision": "Decision", "valid": "Valid test"})


def overload() -> None:
    c = ctx()
    capacity, table, row = c.params["capacity"], c.result.table, _row(c.period)
    page_header("Analysis", "Overload Probability",
                f"P(Xₜ > C): the probability that more than C = {capacity} students arrive in a slot, equivalently "
                f"P(Xₜ ≥ K) with threshold K = {capacity + 1}. Whiskers show how the probability moves across the "
                "confidence interval for λ (research gap G4). <b>Click a bar</b> to inspect a slot.")
    left, right = two_columns()
    with left:
        key = f"overflow_chart_{c.period}"
        chart(viz.overload_by_period(table, c.params["low"], c.params["high"]), key,
              on_select=partial(select_period_from_chart, key), selection_mode="points")
    with right:
        counts = c.result.kpis["risk_counts"]
        figures([(f"{counts['HIGH']}", "high-risk slots"), (f"{counts['MEDIUM']}", "medium-risk slots"),
                 (f"{counts['LOW']}", "low-risk slots")], stacked=True)
        note("How to read it", "Colour is the risk level; dotted lines are the thresholds set in the sidebar.")

    st.markdown("#### 3D risk surface")
    left, right = two_columns()
    with left:
        chart(viz.overload_surface_3d(table, capacity), "overflow_surface")
    with right:
        note("How to read it", "The surface is P(X > C) for every combination of arrival rate λ and capacity C. "
                               f"Each dot is a slot at the current capacity C = {capacity}, coloured by risk. "
                               "Drag to rotate, scroll to zoom.")
        note("What it shows", "Risk rises steeply where λ approaches C: along the ridge a small change in "
                              "capacity or demand moves a slot from Low to High.")

    st.markdown(f"#### Slot {slot(c.period)}")
    left, right = two_columns()
    with left:
        chart(viz.overload_curve(float(row["lambda_hat"]), capacity, slot(c.period)), "overflow_curve")
    with right:
        figures([(pct(row["overload_prob"]), f"P(X > {capacity})"),
                 (f"{pct(row['overload_prob_low'], 1)} to {pct(row['overload_prob_high'], 1)}", "over the λ CI")],
                stacked=True)
        html_block(f"<p>Risk level {pill(row['risk'])}</p>")
        note("How to read it", "The curve answers 'what if capacity changed?' for this slot; the dashed line is "
                               "the current capacity.")
    with st.expander(f"Worked calculation of P(X > {capacity}) for {slot(c.period)}"):
        lam = float(row["lambda_hat"])
        breakdown = overload_breakdown(lam, capacity)
        st.dataframe(breakdown, hide_index=True, height=240, width="stretch", column_config={
            "P(X = x)": st.column_config.NumberColumn(format="%.6f"),
            "P(X ≤ x)": st.column_config.NumberColumn(format="%.6f")})
        st.markdown(f"P(X ≤ {capacity}) = {breakdown['P(X ≤ x)'].iloc[-1]:.6f}, so P(X > {capacity}) = "
                    f"**{overload_probability_manual(lam, capacity):.6f}** by the textbook sum; SciPy's "
                    f"`poisson.sf({capacity}, {lam:.3f})` gives **{overload_probability(lam, capacity):.6f}**.")

    st.markdown("#### Capacity planning")
    target = st.slider("Keep the chance of overload below (%)", 1, 50, int(100 * c.params["low"]),
                       key="plan_target") / 100
    ordered = table.sort_values("period")
    needed = [required_capacity(float(lam), target) for lam in ordered["lambda_hat"]]
    left, right = two_columns()
    with left:
        chart(viz.required_capacity_chart(ordered["period"].tolist(), needed, capacity, target), "capacity_plan")
    with right:
        short = [p for p, n in zip(ordered["period"], needed) if n > capacity]
        note("How to read it", f"Each bar is the smallest capacity that keeps P(overload) below {100 * target:.0f}% "
                               "in that slot. Red bars exceed the current capacity.")
        note("Result", (f"{len(short)} slot(s) need more than the current {capacity}: "
                        + ", ".join(slot(p) for p in short) + f". The peak needs {max(needed)}.")
             if short else f"The current capacity of {capacity} meets the target in every slot.")


@st.fragment
def simulation() -> None:
    c = ctx()
    table, capacity = c.result.table, c.params["capacity"]
    row = _row(c.period)
    lam, p_model = float(row["lambda_hat"]), float(row["overload_prob"])
    page_header("Analysis", "Simulation Check",
                "A probability is a long-run frequency. The fitted model is used to simulate many lunch services, "
                "and the share of days that overflow is compared with the formula.")
    tabs = st.tabs([f"Many days of {slot(c.period)}", "One whole lunch service"])
    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        n_days = c1.select_slider("Simulated days", [100, 250, 500, 1000, 2500, 5000], value=1000, key="mc_days")
        seed = int(c2.number_input("Random seed", 0, 99_999, 7, key="mc_seed"))
        _draws, running = simulate_overload_frequency(lam, capacity, n_days, seed)
        se = float(np.sqrt(p_model * (1 - p_model) / n_days))
        left, right = two_columns()
        with left:
            figures([(pct(p_model), "model P(X > C)"), (pct(float(running[-1])), f"simulated, {n_days} days"),
                     (f"±{100 * 2 * se:.2f} pts", "expected range (2 SE)")])
            chart(viz.monte_carlo_convergence(running, p_model), f"mc_{n_days}_{seed}_{c.period}")
        with right:
            note("How to read it", "Press <b>Play</b>. The line is the share of simulated days so far with more "
                                   "than C arrivals. It wobbles early and settles on the dashed model value: the "
                                   "law of large numbers.")
    with tabs[1]:
        if "day_seed" not in st.session_state:
            st.session_state["day_seed"] = 1
        if st.button("Simulate another day", key="sim_day_button"):
            st.session_state["day_seed"] += 1
        ordered = table.sort_values("period")
        lams = ordered["lambda_hat"].to_numpy()
        counts = np.random.default_rng(st.session_state["day_seed"]).poisson(lams)
        left, right = two_columns()
        with left:
            chart(viz.simulated_day(ordered["period"].tolist(), counts, lams, capacity), "sim_day_chart")
        with right:
            figures([(f"{int((counts > capacity).sum())} of {len(counts)}", "slots overloaded"),
                     (f"{float(ordered['overload_prob'].sum()):.2f}", "expected per day")], stacked=True)
            note("How to read it", "Bars are one simulated day; black ticks are each slot's expected arrivals. "
                                   "Red bars overloaded.")


# ===========================================================================
# RESULTS
# ===========================================================================
def results() -> None:
    c = ctx()
    table = c.result.table.sort_values("period")
    page_header("Results", "Risk Classification and Recommendations",
                f"Each slot labelled Low, Medium or High overload risk at capacity C = {c.params['capacity']}, and "
                "the operational recommendations that follow. Thresholds are project-defined decision rules, "
                "adjustable in the sidebar.")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        chart(viz.risk_distribution(c.result.table), "risk_donut")
    with right:
        rows = "".join(
            f"<tr><td>{slot(r.period)}</td><td>{r.lambda_hat:.1f}</td><td>{pct(r.overload_prob, 1)}</td>"
            f"<td>{pill(r.risk)}</td><td>{html.escape(str(r.gof_decision))}</td></tr>" for r in table.itertuples())
        html_block('<table style="width:100%"><thead><tr><th>Slot</th><th>λ̂</th><th>P(X &gt; C)</th>'
                   f'<th>Risk</th><th>Poisson fit</th></tr></thead><tbody>{rows}</tbody></table>')
    st.markdown("#### Recommendations")
    for rec in c.result.recommendations:
        with st.container(border=True):
            st.markdown(rec)
    st.download_button("Download slot results (CSV)", c.result.table.to_csv(index=False), "canteenflow_results.csv",
                       "text/csv", key="dl_results_page")


@st.fragment
def quietest() -> None:
    c = ctx()
    table, q, capacity = c.result.table, c.result.quiet, c.params["capacity"]
    page_header("Results", "Least-Crowded Time",
                "The slot with the lowest predicted crowd, a statistical test that it is genuinely quieter than "
                "the runner-up, and a planner for your own schedule (research gap G5).")
    html_block(f'<div class="answer">Least predicted crowd: {slot(q["best"])}</div>')
    sure = ""
    if q["runner_up"]:
        sure = (f" Compared with the runner-up, {slot(q['runner_up'])} (λ̂ = {q['runner_up_lambda']:.1f}), the exact "
                f"conditional test gives p = {q['p_value']:.2g}: "
                + ("**significantly quieter**." if q["significant"] else "**not significantly quieter**; treat both "
                                                                         "as equally quiet."))
    st.markdown(f"Expected arrivals **{q['best_lambda']:.1f}** per slot (CI {q['best_ci'][0]:.1f} to "
                f"{q['best_ci'][1]:.1f}) against {c.result.kpis['peak_lambda']:.1f} at the peak; overload "
                f"probability {pct(q['best_overload'])}.{sure}")

    tabs = st.tabs(["Ranking", "Plan your visit", "By weekday"])
    with tabs[0]:
        left, right = two_columns()
        with left:
            chart(viz.quiet_ranking(q["ranking"], q["best"]), "quiet_ranking")
        with right:
            note("How to read it", "Quietest at the top. Whiskers are confidence intervals; strongly overlapping "
                                   "whiskers mean the data cannot separate those slots.")
            if q["best"] == table["period"].max():
                note("Practical caution", "The quietest slot is the last before service ends, when popular dishes may "
                                          "have run out. Use the planner to search within your preferred hours.")
    with tabs[1]:
        periods = sorted(table["period"])
        bounds = [p[:5] for p in periods] + [periods[-1][6:11]]
        c1, c2 = st.columns(2, gap="large")
        earliest, latest = c1.select_slider("I can be at the canteen between", bounds, value=(bounds[0], bounds[-1]),
                                            key="plan_window")
        comfort = c2.slider("I'm comfortable with at most this many arrivals in my slot", 0,
                            int(max(capacity, table["lambda_hat"].max() * 1.5)),
                            int(round(q["best_lambda"] * 1.25)), key="plan_comfort")
        pick = quietest_in_window(table, earliest, latest)
        probs = comfort_probabilities(table.sort_values("period"), comfort)
        left, right = two_columns()
        with left:
            chart(viz.comfort_bars(probs, comfort, pick["period"] if pick is not None else None), "plan_chart")
        with right:
            if pick is None:
                verdict("na", "No complete slot fits between those times. Widen the window.")
            else:
                figures([(slot(pick["period"]), f"best slot, {earliest} to {latest}"),
                         (pct(probs[pick["period"]], 1), f"chance of at most {comfort} arrivals")], stacked=True)
            note("How to read it", f"Each bar is P(X ≤ {comfort}) for a slot. Green marks the best slot in your window.")
    with tabs[2]:
        rates = weekday_rates(c.df, c.params["group_col"])
        left, right = two_columns()
        with left:
            chart(viz.weekday_heatmap(rates), "quiet_weekday")
        with right:
            best = rates.idxmin(axis=1)
            st.dataframe(pd.DataFrame({"Weekday": rates.index, "Quietest slot": best.map(slot).values,
                                       "Mean": [rates.loc[d, s] for d, s in best.items()]}),
                         hide_index=True, width="stretch",
                         column_config={"Mean": st.column_config.NumberColumn(format="%.1f")})
            note("How to read it", "Outlined cells are each weekday's quietest slot. Each cell uses about a fifth "
                                   "of the data, so weekday differences are less certain than the overall ranking.")


# ===========================================================================
# CONCLUSION
# ===========================================================================
def limitations() -> None:
    page_header("Conclusion", "Limitations and Future Scope",
                "What the model leaves out, and how the work can be extended.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Limitations")
        st.markdown(
            "- **Prototype data.** Field collection is in progress; the bundled records are synthetic.\n"
            "- **Arrival risk, not queue length.** Queue length also depends on service rate, number of counters "
            "and queue discipline.\n"
            "- **Constant rate within a slot.** The Poisson model assumes λ does not change inside a slot.\n"
            "- **Independence.** Students arriving in groups cause overdispersion.\n"
            "- **Threshold choice.** Capacity C (or K) must be justified for the chosen canteen and slot length.\n"
            "- **Chosen risk cut-offs.** Low / medium / high thresholds are project decisions.\n"
            "- **External events.** Exam days are modelled when marked in the data; holidays, weather and "
            "timetable changes are not.\n"
            "- **Same peak shape every day.** The regression scales the whole day up or down; it cannot move "
            "the peak to a different slot on special days without slot × day-type interaction terms.\n"
            "- **Log-scale regression.** Least squares on log counts is a syllabus-friendly stand-in for "
            "Poisson regression; means are back-transformed with a smearing factor.")
    with right:
        st.markdown("#### Future scope")
        st.markdown(
            "- **Field data.** Replace prototype records with entry counts collected over m days.\n"
            "- **Queueing extension (M/M/c).** Add service times to estimate queue length and waiting time.\n"
            "- **Done at the second review:** Poisson–Gamma (negative binomial) probabilities for day-to-day "
            "variation, Bayesian same-day updating, peak-time probabilities and a validation backtest.\n"
            "- **Group arrivals.** Compound Poisson models for students arriving together.\n"
            "- **Smooth arrival rate.** Estimate λ(t) continuously as a nonhomogeneous Poisson process "
            "(Leemis, 1991).\n"
            "- **Live data feed.** Run the same-day update automatically from turnstile or camera counts.\n"
            "- **Natural-language and voice interface.** As proposed at the first review: a usability layer that "
            "retrieves the verified results and explains them, without calculating or inventing values.")


def references() -> None:
    c = ctx()
    page_header("Conclusion", "References and Downloads",
                "Studies cited in the literature review and research gaps, statistical method references, and the "
                "data and results used in this dashboard.")
    st.markdown("#### Literature")
    for i, p in enumerate(sorted(PAPERS, key=lambda x: x["reference"]), start=1):
        doi = f" https://doi.org/{p['doi']}" if p["doi"] else ""
        st.markdown(f"{i}. {p['reference']}{doi}")
    st.markdown("#### Statistical methods")
    for i, ref in enumerate(METHOD_REFERENCES, start=len(PAPERS) + 1):
        st.markdown(f"{i}. {ref}")
    st.markdown("#### Downloads")
    preview = c.df.drop(columns=["hour_block"]).assign(date=c.df["date"].dt.date)
    c1, c2 = st.columns(2)
    c1.download_button("Download analysed records (CSV)", preview.to_csv(index=False), "canteen_arrivals_filtered.csv",
                       "text/csv", key="dl_records", width="stretch")
    c2.download_button("Download slot results (CSV)", c.result.table.to_csv(index=False), "canteenflow_results.csv",
                       "text/csv", key="dl_results", width="stretch")


# ===========================================================================
# FORECAST (research gaps G7-G11)
# ===========================================================================
@st.cache_data(show_spinner=False)
def _forecast_basics(df: pd.DataFrame, group_col: str):
    model = fc.fit_rate_model(df, group_col)
    return model, fc.estimate_day_variance(df, model), fc.day_effect_tests(df)


@st.cache_data(show_spinner=False)
def _peak(mu: pd.Series, capacity: int, var_w: float, seed: int = 0) -> dict:
    return fc.peak_time_distribution(mu, capacity, var_w, seed=seed)


@st.cache_data(show_spinner="Running the rolling backtest…")
def _backtest(df: pd.DataFrame, capacity: int, train_days: int, update_after: int, expanding: bool, alpha: float):
    bt = fc.rolling_backtest(df, capacity, train_days, update_after, expanding=expanding)
    return bt, fc.summarize_backtest(bt, alpha)


def _p(value: float) -> str:
    return "n/a" if value != value else ("< 0.001" if value < 0.001 else f"{value:.3f}")


def _forecast_inputs():
    c = ctx()
    if c.params["group_col"] != "time_interval":
        st.info("Forecasting works on individual slots. Set *Slot grouping* to *Individual slots* in the sidebar "
                "(Analysis options).")
        st.stop()
    try:
        return _forecast_basics(c.df, "time_interval")
    except ValueError as err:
        st.warning(f"Not enough data to fit the forecasting model: {err}")
        st.stop()


def syllabus() -> None:
    page_header("Project", "Syllabus Map",
                "Every statistical component of the project and the PAMAT501 module it comes from. The second "
                "review adds forecasting (gaps G7 to G11) using only syllabus methods, so all five taught modules "
                "are now used. <b>Pick modules</b> to filter the table.")
    short = {"M1": "Probability & random variables", "M2": "Distributions", "M3": "Correlation & regression",
             "M4": "Large-sample tests", "M5": "Small-sample & non-parametric tests"}
    counts = {m: sum(m in row[2] for row in SYLLABUS_MAP) for m in SYLLABUS_MODULES}
    figures([(f"{n}", f"components use {m}, {SYLLABUS_MODULES[m]}") for m, n in counts.items()])
    chosen = st.pills("Modules", list(SYLLABUS_MODULES), selection_mode="multi", default=list(SYLLABUS_MODULES),
                      format_func=lambda m: f"{m} {short[m]}", key="syl_filter")
    chosen = chosen or list(SYLLABUS_MODULES)
    rows = [{"Component": comp, "Syllabus topics": topics, "Modules": ", ".join(mods), "Dashboard page": where,
             "Gap": gap} for comp, topics, mods, where, gap in SYLLABUS_MAP if set(mods) & set(chosen)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(36 * len(rows) + 40, 920),
                 column_config={"Component": st.column_config.TextColumn(width="large"),
                                "Syllabus topics": st.column_config.TextColumn(width="large"),
                                "Modules": st.column_config.TextColumn(width="small"),
                                "Dashboard page": st.column_config.TextColumn(width="medium"),
                                "Gap": st.column_config.TextColumn(width="small")})
    st.caption("Module 6 (contemporary issues, guest lectures) has no statistical content to map. Methods from the "
               "forecasting papers that lie outside the syllabus (factor models, MCMC, integer-GARCH, machine "
               "learning) are cited as context only and are not used.")
    _link("forecast", "Open the peak-time forecast")


def forecast() -> None:
    c = ctx()
    model, disp, effects = _forecast_inputs()
    capacity, var_w = c.params["capacity"], disp["var_w"]
    page_header("Forecast", "Peak-Time Forecast",
                "Predicts the overload probability of every slot for a chosen kind of day, how likely each slot "
                "is to be the day's peak, and how today's early counts change the rest of the forecast "
                "(research gaps G7, G8 and G9). Rotate any 3D chart by dragging; scroll to zoom.")
    c1, c2, c3 = st.columns([1, 1, 1.2], vertical_alignment="bottom")
    weekday = c1.selectbox("Weekday", model.weekdays, key="fc_weekday")
    day_type = c2.selectbox("Day type", model.day_types, key="fc_daytype",
                            help="Exam days come from the day_type column; field CSVs without it are all Regular.")
    mu = model.predict(weekday, day_type)
    p_over = pd.Series(fc.overload_probability_mixed(mu.to_numpy(), capacity, var_w), index=mu.index)
    pk = _peak(mu, capacity, var_w)
    c3.markdown(f"Model fitted on **{model.n}** slot counts from **{c.df['date'].nunique()}** days; "
                f"day-to-day variation v̂ = {var_w:.4f}.")

    html_block(f'<div class="answer">Most likely peak: {slot(pk["most_likely_peak"])}</div>')
    top = " or ".join(slot(p) for p in sorted(pk["top_two"]))
    st.markdown(f"On a **{day_type.lower()} {weekday}** the busiest slot is {slot(pk['most_likely_peak'])} with "
                f"probability **{pct(pk['p_most_likely'], 0)}**; the peak is {top} with probability "
                f"**{pct(pk['p_top_two'], 0)}**. The chance that at least one slot exceeds C = {capacity} is "
                f"**{pct(pk['p_any_overload'], 0)}**.")
    figures([(pct(pk["p_most_likely"], 0), f"P(peak = {slot(pk['most_likely_peak'])})"),
             (pct(pk["p_any_overload"], 0), "P(at least one overload)"),
             (f"{pk['expected_overloaded_slots']:.1f}", "expected overloaded slots"),
             (f"{mu.max():.1f}", "highest predicted λ"),
             (f"{pk['peak_count_quantiles'][0]:.0f}–{pk['peak_count_quantiles'][2]:.0f}", "90% range of the peak count")])

    tabs = st.tabs(["3D risk surface", "Peak probability", "3D canteen view", "Same-day update", "Regression model"])
    with tabs[0]:
        grid = pd.DataFrame({w: fc.overload_probability_mixed(model.predict(w, day_type).to_numpy(), capacity, var_w)
                             for w in model.weekdays}, index=model.periods).T
        left, right = two_columns()
        with left:
            chart(viz.forecast_surface_3d(grid, weekday), "fc_surface")
        with right:
            note("How to read it", f"Height and colour are the predicted P(X > {capacity}) for every weekday and "
                                   f"slot on a {day_type.lower()} day. The red line is the weekday selected above.")
            note("Why it is a forecast", "Each value comes from the regression rate for that weekday and day type, "
                                         "not from an average of past days, and it includes day-to-day "
                                         "variation (Poisson–Gamma tail).")
    with tabs[1]:
        left, right = two_columns()
        with left:
            chart(viz.peak_probability_bars(pk["p_peak"]), "fc_peak")
        with right:
            note("How to read it", f"Share of {pk['n_sims']:,} simulated lunch periods in which each slot had the "
                                   "most arrivals. Close neighbours split the probability: the peak time is "
                                   "uncertain even when the rates are known.")
            note("Method", "Each simulated day draws a busyness factor W ~ Gamma(mean 1, variance v̂), then "
                           "Xₜ ~ Poisson(W·λₜ) for every slot (Modules 1 and 2).")
        risk = [classify_risk(p, c.params["low"], c.params["high"]) for p in p_over]
        st.dataframe(pd.DataFrame({"Slot": [slot(p) for p in mu.index], "Predicted λ": mu.to_numpy(),
                                   "P(X > C)": 100 * p_over.to_numpy(), "Risk": [ui.RISK_WORD[r] for r in risk],
                                   "P(peak)": 100 * pk["p_peak"].to_numpy()}),
                     hide_index=True, width="stretch",
                     column_config={"Predicted λ": st.column_config.NumberColumn(format="%.2f"),
                                    "P(X > C)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                                                max_value=100),
                                    "P(peak)": st.column_config.NumberColumn(format="%.1f%%")})
    with tabs[2]:
        _canteen_view(mu, p_over, capacity, var_w)
    with tabs[3]:
        _same_day_update(mu, capacity, var_w)
    with tabs[4]:
        _regression_view(model, effects, c.params["alpha"])


@st.fragment
def _canteen_view(mu: pd.Series, p_over: pd.Series, capacity: int, var_w: float) -> None:
    if "fc_scene_seed" not in st.session_state:
        st.session_state["fc_scene_seed"] = 0
    left, right = two_columns()
    rng = np.random.default_rng(st.session_state["fc_scene_seed"])
    w = rng.gamma(1 / var_w, var_w) if var_w > fc.MIN_VARIANCE else 1.0
    counts = rng.poisson(w * mu.to_numpy())
    with left:
        chart(viz.canteen_scene_3d(list(mu.index), counts, mu.to_numpy(), capacity, p_over.to_numpy(),
                                   seed=st.session_state["fc_scene_seed"]),
              f"fc_scene_{st.session_state['fc_scene_seed']}")
    with right:
        if st.button("Simulate another day", width="stretch", key="fc_scene_new"):
            st.session_state["fc_scene_seed"] += 1
            st.rerun(scope="fragment")
        figures([(f"{w:.3f}", "this day's busyness factor W"),
                 (str(int((counts > capacity).sum())), "slots over capacity"),
                 (slot(mu.index[int(np.argmax(counts))]), "busiest slot on this day")], stacked=True)
        note("How to read it", f"Press Play (or drag the slider) to step through the lunch period. Each dot is a "
                               f"simulated student: the first {capacity} of a slot queue at the two counters "
                               "(teal); the rest wait near the entrance as overflow (red).")


@st.fragment
def _same_day_update(mu: pd.Series, capacity: int, var_w: float) -> None:
    periods = list(mu.index)
    if len(periods) < 2:
        st.info("Same-day updating needs at least two time slots per day.")
        return
    k = st.slider("Counts known up to and including slot", 1, len(periods) - 1, min(2, len(periods) - 1),
                  format="%d", key="fc_known", help="The slots already observed today.")
    st.markdown(f"Today's counts so far (forecast in brackets). Known slots: "
                f"{', '.join(slot(p) for p in periods[:k])}.")
    cols = st.columns(min(k, 6))
    observed = {}
    for i, p in enumerate(periods[:k]):
        observed[p] = cols[i % len(cols)].number_input(f"{p[:5]} ({mu[p]:.0f})", 0, 1000, int(round(mu[p] * 1.15)),
                                                       key=f"fc_obs_{p}")
    table = fc.same_day_update(mu, pd.Series(observed), capacity, var_w)
    ratio = sum(observed.values()) / mu.iloc[:k].sum()
    figures([(f"{100 * ratio:.0f}%", "arrivals so far, % of forecast"),
             (f"{table['w_mean'].iloc[0]:.3f}", "posterior mean of W"),
             (f"± {table['w_sd'].iloc[0]:.3f}", "posterior s.d. of W")])
    left, right = two_columns()
    with left:
        chart(viz.update_before_after(table), "fc_update_bars")
    with right:
        note("Bayes theorem", "Prior W ~ Gamma(a, a) with a = 1/v̂. After observing counts x with forecasts λ, "
                              "the posterior is Gamma(a + Σx, a + Σλ), and each later slot follows a negative "
                              "binomial with that posterior (Modules 1 and 2).")
        note("Why the change is small or large", "When days differ little (small v̂), the prior is strong and "
                                                 "a busy morning moves the forecast only a little. With field "
                                                 "data that vary more from day to day, the same counts move it "
                                                 "much further.")
    left, right = two_columns()
    with left:
        chart(viz.update_surface_3d(mu, capacity, var_w, k), f"fc_update_surface_{k}")
    with right:
        note("How to read it", "Every possible morning at once: the y-axis is how busy the known slots were "
                               "relative to the forecast, the height is the updated P(X > C) of each remaining "
                               "slot. The dashed line is a day exactly as forecast.")


def _regression_view(model, effects: dict, alpha: float) -> None:
    left, right = two_columns()
    with left:
        chart(viz.coefficient_chart(model.coefficients), "fc_coef")
    with right:
        figures([(f"{model.r_squared:.3f}", "R² (log scale)"), (f"{model.adj_r_squared:.3f}", "adjusted R²"),
                 (f"{model.n}", "observations"), (f"{model.dof}", "residual df")], stacked=False)
        note("Model", "log(x + 0.5) = b₀ + slot + weekday + day type + e, fitted by least squares (Module 3). Each "
                      "effect is shown as a multiplier on expected arrivals with a 95% interval: teal raises arrivals, "
                      f"blue lowers them, grey is not significant at α = {alpha} (t-test, Module 5). Means are back-transformed with the smearing factor "
                      f"{model.smear:.3f} (Duan, 1983).")
    coef = model.coefficients.assign(p_value=model.coefficients["p_value"].map(_p))
    st.dataframe(coef.rename(columns={"term": "Term", "estimate": "Estimate", "std_error": "Std. error", "t": "t",
                                      "p_value": "p-value", "multiplier": "Multiplier"}),
                 hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.4f")
                                for k in ("Estimate", "Std. error", "t", "Multiplier")})
    st.markdown("#### Do the day effects matter?")
    wk, dt = effects.get("weekday"), effects.get("day_type")
    if wk:
        sig = wk["anova_p"] < alpha
        verdict("ok" if sig else "na",
                f"<b>Weekday (regular days, daily totals):</b> one-way ANOVA F = {wk['anova_f']:.2f}, "
                f"p = {_p(wk['anova_p'])}; Kruskal-Wallis H = {wk['kruskal_h']:.2f}, p = {_p(wk['kruskal_p'])}. "
                + ("Weekdays differ, so a weekday term is justified." if sig else
                   "No significant weekday difference; the weekday terms add little."))
    if dt:
        sig = dt["t_p"] < alpha
        verdict("ok" if sig else "na",
                f"<b>Exam vs regular days:</b> mean daily total {dt['special_mean']:.1f} vs {dt['regular_mean']:.1f} "
                f"({dt['n_special']} vs {dt['n_regular']} days). Welch t = {dt['t']:.2f}, p = {_p(dt['t_p'])}; "
                f"Mann-Whitney U = {dt['u']:.0f}, p = {_p(dt['u_p'])}. "
                + ("Exam days differ, so the day-type term is justified." if sig else "No significant difference."))
    elif len(model.day_types) == 1:
        st.caption("All days are of one type. Add a day_type column (e.g. Regular / Exam) to field data to "
                   "estimate special-day effects.")


def validation() -> None:
    c = ctx()
    model, disp, _ = _forecast_inputs()
    capacity, alpha = c.params["capacity"], c.params["alpha"]
    n_days = c.df["date"].nunique()
    page_header("Forecast", "Forecast Validation",
                "Each day is forecast using only the days before it, then compared with what actually happened "
                "(research gap G11). The regression model is tested against the per-slot average used at the "
                "first review. Also: the overdispersion check (G10) and day-to-day dependence.")
    if n_days < 16:
        st.warning("At least 16 days are needed for a backtest. Widen the filters in the sidebar.")
        st.stop()
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
    train = c1.slider("Days before the first forecast", 10, min(60, n_days - 5), min(30, n_days - 5), key="bt_train")
    window = c2.radio("Training window", ["All earlier days", f"Last {train} days only"], horizontal=True,
                      key="bt_window")
    n_slots = c.df["time_interval"].nunique()
    upd = (c3.slider("Same-day update after slot", 1, n_slots - 1, min(2, n_slots - 1), key="bt_update")
           if n_slots > 2 else 1)
    try:
        bt, s = _backtest(c.df, capacity, train, upd, window.startswith("All"), alpha)
    except ValueError as err:
        st.warning(str(err))
        st.stop()

    better = s["brier_skill"] > 0
    figures([(f"{s['n_days']}", "days forecast"), (f"{s['brier_model']:.4f}", "Brier score, model"),
             (f"{s['brier_base']:.4f}", "Brier score, per-slot average"),
             (f"{100 * s['brier_skill']:+.1f}%", "Brier skill vs average"),
             (pct(s["coverage"], 1), "95% interval coverage"),
             (pct(s["hit_rate_model"], 0), f"peak slot hit rate (chance {pct(s['hit_chance'], 0)})")])
    paired = s["tests"].iloc[2]
    st.markdown(
        f"Over {s['n_days']} forecast days the model's mean absolute error is **{s['mae_model']:.2f}** arrivals per "
        f"slot against **{s['mae_base']:.2f}** for the per-slot average; its probability forecasts are "
        f"{'better' if better else 'not better'} by the Brier skill score ({100 * s['brier_skill']:+.1f}%). "
        f"The paired t-test gives p = {_p(paired['p_value'])}, so the difference is "
        f"{'significant' if paired['p_value'] < alpha else '**not statistically significant** on this data'}. "
        + ("" if paired["p_value"] < alpha else "With the mild weekday and exam effects in the prototype data, the "
           "per-slot average is hard to beat (as Ibrahim et al., 2016, report for long horizons); the "
           "model's value is in calibrated probabilities, day-type forecasts and same-day updating."))

    tabs = st.tabs(["Hypothesis tests", "3D forecast vs actual", "Calibration", "Daily errors",
                    "Overdispersion (G10)", "Dependence"])
    with tabs[0]:
        t = s["tests"]
        st.dataframe(pd.DataFrame({
            "Check": t["check"], "Method": t["method"], "Module": t["module"], "Statistic": t["statistic"],
            "p-value": t["p_value"].map(_p), "Result": t["result"],
            "Supports the model": t["supports_model"].map({True: "Yes", False: "No", None: "n/a"})}),
            hide_index=True, width="stretch")
        note("How to read it", f"α = {alpha}. For coverage and the PIT tests the model is supported when H₀ is "
                               "<i>not</i> rejected (the forecasts behave as claimed); for the hit rate and the "
                               "model-vs-average tests it is supported when H₀ <i>is</i> rejected.")
        if "brier_updated" in s:
            gain = 1 - s["brier_updated"] / s["brier_not_updated"]
            st.markdown(f"**Same-day update (G9):** for slots after the first {upd}, the Brier score is "
                        f"{s['brier_updated']:.4f} with the update and {s['brier_not_updated']:.4f} without "
                        f"({100 * gain:+.1f}%).")
        st.markdown(f"**Correlation of predicted and actual counts (Module 3):** Pearson r = {s['pearson_r']:.3f}, "
                    f"Spearman ρ = {s['spearman_rho']:.3f} (both p {('< 0.001' if s['pearson_p'] < 0.001 else '= ' + _p(s['pearson_p']))}).")
    with tabs[1]:
        left, right = two_columns()
        with left:
            chart(viz.backtest_surface_3d(bt.slots), "bt_surface")
        with right:
            note("How to read it", "The surface is what actually arrived on each test day; the orange points are "
                                   "the forecasts made the evening before. Points that sit on the surface are "
                                   "accurate forecasts. Drag to rotate.")
    with tabs[2]:
        left, right = st.columns(2, gap="large")
        with left:
            chart(viz.reliability_chart(s["reliability_model"], s["reliability_base"]), "bt_rel")
            note("Reliability", "Forecasts grouped by probability against how often overload actually happened. "
                                "Points on the diagonal are calibrated.")
        with right:
            chart(viz.pit_histogram(s["pit_counts"]), "bt_pit")
            note("PIT", "Where each actual count fell in its forecast distribution. A flat histogram means the "
                        "whole distribution, not only the mean, is right (Kolmogorov–Smirnov and chi-square tests).")
    with tabs[3]:
        left, right = two_columns()
        with left:
            chart(viz.daily_error_comparison(bt.days), "bt_daily")
        with right:
            g = s["mae_by_group"]
            st.dataframe(g.rename(columns={"group": "Days", "days": "n", "mae_model": "MAE model",
                                           "mae_base": "MAE average"}), hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.2f")
                                        for k in ("MAE model", "MAE average")})
            note("How to read it", "Each point is one test day; below the diagonal the model was more accurate. "
                                   "The table splits days by type, where the weekday and exam terms should help.")
    with tabs[4]:
        _overdispersion_view(disp, capacity, alpha)
    with tabs[5]:
        corr = fc.consecutive_slot_correlation(c.df, model)
        runs = fc.daily_residual_runs(c.df, model)
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Carry-over between consecutive slots** (Spearman ρ of observed ÷ predicted)")
            st.dataframe(corr.assign(pair=corr["pair"].map(slot), p_value=corr["p_value"].map(_p))
                         .rename(columns={"pair": "Slots", "rho": "ρ", "p_value": "p-value"}), hide_index=True,
                         width="stretch", column_config={"ρ": st.column_config.NumberColumn(format="%.3f")})
            note("Why it matters", "Positive ρ means a busier-than-forecast slot tends to be followed by another, "
                                   "which is what makes the same-day update work (Ibrahim et al., 2016, Table 2).")
        with right:
            st.markdown("**Do busy days follow busy days?** (runs test on daily residual signs)")
            verdict("na" if runs["p_value"] < alpha else "ok",
                    f"{runs['runs']} runs against {runs['expected']:.1f} expected ({runs['n_pos']} days above "
                    f"forecast, {runs['n_neg']} below); z = {runs['z']:.2f}, p = {_p(runs['p_value'])}. "
                    + ("Days are not independent: a day-to-day term would help." if runs["p_value"] < alpha else
                       "No evidence that busy days cluster; treating days as independent is reasonable."))
            note("Method", "Wald–Wolfowitz run test (Module 5). Weinberg et al. (2007) and Aldor-Noiman et al. "
                           "(2009) found strong day-to-day dependence in call centres; this checks it for the canteen.")


def _overdispersion_view(disp: dict, capacity: int, alpha: float) -> None:
    c = ctx()
    table = c.result.table
    sig = disp["p_value"] < alpha
    verdict("na" if sig else "ok",
            f"Pearson χ² = {disp['pearson_chi2']:.1f} on {disp['dof']} df (ratio {disp['dispersion_ratio']:.3f}), "
            f"p = {_p(disp['p_value'])}. "
            + ("Counts vary more than Poisson allows: use the Poisson–Gamma probabilities." if sig else
               "No significant overdispersion: the plain Poisson probabilities are adequate, and the "
               "Poisson–Gamma correction below is small."))
    figures([(f"{disp['var_w']:.4f}", "v̂ = Var(W), day-to-day variance"),
             (f"{100 * disp['cv_w']:.1f}%", "day-to-day variation (CV of W)")])
    p_mixed = fc.overload_probability_mixed(table["lambda_hat"].to_numpy(), capacity, disp["var_w"])
    left, right = two_columns()
    with left:
        chart(viz.poisson_vs_mixed_bars(table["period"].tolist(), table["overload_prob"], p_mixed), "bt_mixed")
    with right:
        st.latex(r"P(X = x) = \int_0^\infty \frac{e^{-w\lambda}(w\lambda)^x}{x!}\, g(w)\,dw,\quad "
                 r"W \sim \mathrm{Gamma}\!\left(\tfrac1v, \tfrac1v\right)")
        st.latex(r"E[X] = \lambda,\qquad \mathrm{Var}(X) = \lambda + v\lambda^2")
        note("Method", "Theorem of total probability over the day's busyness factor W gives a negative binomial "
                       "(Modules 1 and 2). v is estimated by moments from the Pearson statistic, "
                       "v̂ = (χ² − df) / Σλ (Cameron & Trivedi, 1990).")
