"""
report_pages.py
---------------
One function per dashboard page. app.py registers them with st.navigation and
fills ui.CTX before the selected page runs.

Project      Overview, Literature Review, Research Gaps, Methodology
Analysis     Data & EDA, Arrival Rate, Goodness of Fit, Overload Probability, Simulation Check
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

import visualization as viz
from analysis import DEMAND_METHODS, comfort_probabilities, quietest_in_window, weekday_rates
from goodness_of_fit import MIN_EXPECTED_FREQUENCY, chi_square_gof, dispersion_test
from literature import GAPS, METHOD_REFERENCES, PAPERS, paper
from poisson_model import (overload_breakdown, overload_probability, overload_probability_manual,
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
        f"least crowded is **{slot(q['best'])}** (λ̂ = {q['best_lambda']:.1f}).")
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
               '<i>→</i><span>Least-crowded time</span></div>')
    st.markdown("#### Objectives")
    objectives = [
        ("Collect and organise arrival data", "Arrival counts at fixed 5-, 10- or 15-minute intervals across m days, "
                                              "validated and cleaned.", "data"),
        ("Estimate time-specific rates", "λ̂ₜ with exact confidence intervals, and a test of Poisson suitability.",
         "rate"),
        ("Calculate overload probability", "P(Xₜ > C) for a chosen capacity, with its uncertainty.", "overload"),
        ("Identify useful time windows", "Low / medium / high risk labels and the least-crowded slot.", "quietest"),
    ]
    cols = st.columns(2, gap="medium")
    for i, (title, text, page_key) in enumerate(objectives):
        with cols[i % 2], st.container(border=True):
            st.markdown(f"**{i + 1}. {title}**  \n{text}")
            _link(page_key, "Open the analysis")


def literature() -> None:
    page_header("Project", "Literature Review",
                "Ten studies reviewed: the base paper plus nine more, from canteen modelling, queueing, "
                "arrival-process statistics, time-varying demand and campus crowd sensing. <b>Select a row</b> "
                "to read a study's focus, method, findings and what it leaves open for this project.")
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
               if selected["doi"] else "No DOI (CNKI record)")
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
        "Slot-level probabilities from simple arrival counts |")
    _link("gaps", "Continue to the research gaps")


def gaps() -> None:
    page_header("Project", "Research Gaps",
                "Six gaps identified from the survey. G1 was stated at the first review; G2 to G6 come from the "
                "extended review. Each is a <b>focus and application</b> gap, not a claim that Poisson "
                "modelling of arrivals is new. Pick a gap to see what it means, the evidence for it, and where "
                "this project addresses it.")
    labels = [f'{g["id"]} {g["title"].split()[0]}' for g in GAPS]
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
                "The statistical framework in seven steps, the formulas behind each test, and an interactive "
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
                    "Goodness of fit", "Comparing two slots"])
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
            "- Attributes: date, day, time slot, arrival count (optional: period, capacity).\n"
            "- CSV columns: `date, time_interval, arrivals`, e.g. `2026-01-05, 12:30-12:40, 14`.\n"
            "- Missing counts are dropped, not filled in: imputing the mean would shrink the variance and bias the "
            "dispersion test.")
        if c.synthetic:
            st.info("Field data collection is in progress. Until it is complete, the analysis uses synthetic "
                    "prototype records generated with a seeded random generator; they are not observed results.")

    tabs = st.tabs(["Distribution by slot", "Weekday profile", "Day × slot heatmap", "Records"])
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
            "- **External events.** Exams, holidays, weather and timetable changes shift arrival patterns.")
    with right:
        st.markdown("#### Future scope")
        st.markdown(
            "- **Field data.** Replace prototype records with entry counts collected over m days.\n"
            "- **Queueing extension (M/M/c).** Add service times to estimate queue length and waiting time.\n"
            "- **Overdispersed counts.** Negative binomial or compound Poisson models for group arrivals "
            "(Cameron & Trivedi, 1990).\n"
            "- **Smooth arrival rate.** Estimate λ(t) continuously as a nonhomogeneous Poisson process "
            "(Leemis, 1991).\n"
            "- **Live updating.** Bayesian Gamma–Poisson updating of λ from turnstile or camera counts.\n"
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
