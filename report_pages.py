"""
report_pages.py
---------------
One function per dashboard page. app.py registers them with st.navigation and
fills ui.CTX before the selected page runs.

Every page follows the same template: the title is a question, the answer sits
directly underneath, then at most three tabs of charts. Each chart has a
one-line caption and an "About this chart" popover with the details.

Plan       When to go (student view, plain language)
Analysis   Arrival pattern, Model check, Overload risk, Peak forecast, Forecast check
Report     About the project, Methodology, Limitations & references
"""

from __future__ import annotations

import html
from datetime import date
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import forecast as fc
import ui
import visualization as viz
from analysis import contiguous_windows, window_label
from goodness_of_fit import MIN_EXPECTED_FREQUENCY, chi_square_gof, dispersion_test
from literature import GAPS, METHOD_REFERENCES, PAPERS, SYLLABUS_MAP, SYLLABUS_MODULES, paper
from poisson_model import (classify_risk, overload_breakdown, overload_probability, overload_probability_manual,
                           required_capacity, simulate_overload_frequency)
from ui import chart_block, ctx, figures, html_block, page_header, pct, pill, slot, slot_picker, verdict
from utils import DAY_NAMES

TEAM = [("Mayur Krishna", "26MCA0191"), ("Faraz Akhtar Khan", "26MCA0197"), ("Malhar Bhosale", "26MCA0199")]
PLAIN_RISK = {"LOW": "usually fine", "MEDIUM": "sometimes crowded", "HIGH": "often overcrowded"}


def _row(period: str) -> pd.Series:
    return ctx().result.table.set_index("period").loc[period]


def _link(page_key: str, label: str) -> None:
    page = ctx().pages.get(page_key)
    if page is not None:
        st.page_link(page, label=label, icon=":material/arrow_forward:")


def _p(value: float) -> str:
    return "n/a" if value != value else ("< 0.001" if value < 0.001 else f"{value:.3f}")


def _plain_pct(p: float) -> str:
    """Whole percentages for the student page; tiny chances read 'under 1%' rather than a misleading 0%."""
    return "under 1%" if p < 0.01 else f"{100 * p:.0f}%"


def _title(kicker: str, question: str) -> None:
    html_block(f'<p class="page-kicker">{html.escape(kicker)}</p><div class="page-title" role="heading" '
               f'aria-level="1">{html.escape(question)}</div>')


# ---------------------------------------------------------------------------
# Cached forecasting results (shared by the Plan and Forecast pages)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _forecast_basics(df: pd.DataFrame):
    model = fc.fit_rate_model(df, "time_interval")
    return model, fc.estimate_day_variance(df, model), fc.day_effect_tests(df)


@st.cache_data(show_spinner=False)
def _peak(mu: pd.Series, capacity: int, var_w: float, seed: int = 0) -> dict:
    return fc.peak_time_distribution(mu, capacity, var_w, seed=seed)


@st.cache_data(show_spinner="Testing the forecasts on past days…")
def _backtest(df: pd.DataFrame, capacity: int, train_days: int, update_after: int, expanding: bool, alpha: float):
    bt = fc.rolling_backtest(df, capacity, train_days, update_after, expanding=expanding)
    return bt, fc.summarize_backtest(bt, alpha)


def _forecast_inputs():
    try:
        return _forecast_basics(ctx().df)
    except ValueError as err:
        st.warning(f"Not enough data to make a forecast: {err} Widen the filters under Advanced settings.")
        st.stop()


def _day_pickers(model, key: str) -> tuple[str, str]:
    """Weekday and exam-week controls shared by the Plan and Forecast pages."""
    weekdays = model.weekdays
    today = DAY_NAMES[date.today().weekday()]
    if st.session_state.get(f"{key}_day") not in weekdays:
        st.session_state[f"{key}_day"] = today if today in weekdays else weekdays[0]
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    weekday = c1.segmented_control("Day", weekdays, key=f"{key}_day", required=True) or weekdays[0]
    exam = False
    if "Exam" in model.day_types:
        exam = c2.toggle("Exam week", key=f"{key}_exam", help="Exam weeks are quieter at lunch.")
    return weekday, ("Exam" if exam else fc.REGULAR)


# ===========================================================================
# PLAN: the student view
# ===========================================================================
def plan() -> None:
    c = ctx()
    model, disp, _ = _forecast_inputs()
    capacity, var_w = c.params["capacity"], disp["var_w"]
    _title("Plan your visit", "When should I go to the canteen?")
    weekday, day_type = _day_pickers(model, "plan")
    mu = model.predict(weekday, day_type)
    periods = list(mu.index)
    probs = fc.overload_probability_mixed(mu.to_numpy(), capacity, var_w)
    risks = [classify_risk(p, c.params["low"], c.params["high"]) for p in probs]
    pk = _peak(mu, capacity, var_w)
    best = mu.idxmin()
    avoid = (contiguous_windows(periods, [r == "HIGH" for r in risks])
             or contiguous_windows(periods, [r == "MEDIUM" for r in risks]))
    avoid_text = ", ".join(window_label(w) for w in avoid) if avoid else "no slot"
    day_words = f"{weekday} in an exam week" if day_type == "Exam" else f"a usual {weekday}"

    html_block(f'<div class="answer-box">On {day_words}, the canteen is busiest at '
               f'<b>{slot(pk["most_likely_peak"])}</b>. Try to avoid <b>{avoid_text}</b>. '
               f'The quietest time is <b>{slot(best)}</b>.</div>')
    figures([(slot(pk["most_likely_peak"]), f"busiest time ({pct(pk['p_most_likely'], 0)} chance)"),
             (avoid_text, "often overcrowded"),
             (slot(best), f"quietest time (about {mu[best]:.0f} students)"),
             (pct(pk["p_any_overload"], 0), "chance it gets overcrowded at some point")])

    chart_block(viz.plan_bars(periods, probs, risks, best, c.params["low"], c.params["high"]), "plan_bars",
                f"Chance that more than {capacity} students arrive in each {c.minutes}-minute slot "
                "(more than the counters can serve).",
                [("Where these numbers come from", "A model fitted to past arrival counts predicts how many "
                  f"students arrive in each slot on {day_words}, allowing for some days being busier than "
                  "others. See <i>Peak forecast</i> for the details and <i>Forecast check</i> for how "
                  "well these predictions did on past days."),
                 ("What overcrowded means", f"More than {capacity} students arriving in one slot: the "
                  "capacity set in the sidebar.")])

    st.markdown("#### Find a time that suits you")
    bounds = [p[:5] for p in periods] + [periods[-1][6:11]]
    start, end = st.select_slider("I can be at the canteen between", bounds, value=(bounds[0], bounds[-1]),
                                  key="plan_window")
    inside = [i for i, p in enumerate(periods) if p[:5] >= start and p[6:11] <= end]
    if not inside:
        verdict("na", "No complete slot fits between those times. Widen the window.")
    else:
        i = min(inside, key=lambda j: (probs[j], mu.iloc[j]))
        verdict("ok" if risks[i] == "LOW" else "na",
                f"Go at <b>{slot(periods[i])}</b>: about {mu.iloc[i]:.0f} students arrive then, and the chance of "
                f"overcrowding is {_plain_pct(probs[i])} ({PLAIN_RISK[risks[i]]}).")

    st.markdown("#### What the lunch hour looks like")
    _canteen_view(mu, pd.Series(probs, index=periods), capacity, var_w)

    if c.synthetic:
        st.caption("These results use synthetic prototype data while field counts are being collected.")
    _link("forecast", "How the forecast is made")


@st.fragment
def _canteen_view(mu: pd.Series, p_over: pd.Series, capacity: int, var_w: float) -> None:
    if "scene_seed" not in st.session_state:
        st.session_state["scene_seed"] = 0
    rng = np.random.default_rng(st.session_state["scene_seed"])
    w = rng.gamma(1 / var_w, var_w) if var_w > fc.MIN_VARIANCE else 1.0
    counts = rng.poisson(w * mu.to_numpy())
    chart_block(viz.canteen_scene_3d(list(mu.index), counts, mu.to_numpy(), capacity, p_over.to_numpy(),
                                     seed=st.session_state["scene_seed"]),
                f"scene_{st.session_state['scene_seed']}",
                f"One simulated day. Press Play to step through the lunch hour; drag to rotate. Teal: students "
                f"queuing at the two counters. Red: students beyond the capacity of {capacity}.",
                [("What this day shows", f"{int((counts > capacity).sum())} of {len(counts)} slots over capacity; "
                  f"busiest slot {slot(mu.index[int(np.argmax(counts))])}. Every simulated day differs, as real "
                  "days do."),
                 ("How it is simulated", "The day's busyness factor W is drawn from a Gamma distribution, then each "
                  "slot's arrivals from a Poisson distribution with the predicted rate times W.")])
    if st.button("Simulate another day", key="scene_new", icon=":material/refresh:"):
        st.session_state["scene_seed"] += 1
        st.rerun(scope="fragment")


# ===========================================================================
# ANALYSIS
# ===========================================================================
def arrivals() -> None:
    c = ctx()
    table, k, q = c.result.table, c.result.kpis, c.result.quiet
    page_header("Analysis", "When do students arrive?",
                f"Arrivals peak at <b>{slot(k['peak_period'])}</b> with about <b>{k['peak_lambda']:.0f}</b> students "
                f"per {c.minutes} minutes, and are lowest at <b>{slot(q['best'])}</b> (about {q['best_lambda']:.0f}).",
                f"{k['n_days']} days of counts, {k['total_observations']:,} slot observations. Source: "
                f"{html.escape(c.raw_label)}.")
    if c.params["group_col"] == "hour_block":
        st.info("Hourly view: slots are pooled into hours, which hides the rise and fall inside each hour (research "
                "gap G3). Switch back to individual slots under Advanced settings.")
    tabs = st.tabs(["Arrivals per slot", "Day-to-day variation", "Data and cleaning"])
    with tabs[0]:
        key = f"rate_chart_{c.period}"
        chart_block(viz.lambda_by_period(table, c.params["capacity"], c.period), key,
                    "Average arrivals per slot (λ̂) with its confidence interval. Click a bar to inspect that slot.",
                    [("How to read it", "Bar height is λ̂, the mean count across days; the whisker is the range of "
                      "rates consistent with the data (exact Garwood interval). Bars above the dashed line exceed "
                      "the capacity on an average day."),
                     ("Colours", f"Demand level, ranked by λ̂: {html.escape(c.result.demand_info['rule'])}")],
                    on_select=partial(ui.select_period_from_chart, key), selection_mode="points")
        left, right = st.columns([1, 2], gap="large", vertical_alignment="center")
        with left:
            period = slot_picker("pick_rate")
        row = _row(period)
        with right:
            st.latex(rf"\hat\lambda = \frac{{\sum x_i}}{{m}} = \frac{{{int(row['total_arrivals'])}}}{{{int(row['n'])}}}"
                     rf" = {row['lambda_hat']:.3f},\quad {100 * (1 - c.params['alpha']):.0f}\%\ \text{{CI}} = "
                     rf"[{row['lambda_ci_low']:.2f},\ {row['lambda_ci_high']:.2f}]")
        if q["runner_up"]:
            verdict("ok" if q["significant"] else "na",
                    f"<b>Is the quietest slot really quieter?</b> {slot(q['best'])} (λ̂ = {q['best_lambda']:.1f}) "
                    f"against {slot(q['runner_up'])} (λ̂ = {q['runner_up_lambda']:.1f}): exact conditional test "
                    f"p = {q['p_value']:.2g}, so it is "
                    + ("<b>significantly quieter</b>." if q["significant"] else "<b>not significantly quieter</b>; "
                                                                                "treat both as equally quiet."))
        with st.expander("Summary statistics for every slot"):
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
    with tabs[1]:
        group_col = c.params["group_col"]
        view = st.segmented_control("View", ["By slot", "By weekday", "Every day"], default="By slot",
                                    key="var_view", label_visibility="collapsed") or "By slot"
        if view == "By slot":
            chart_block(viz.slot_boxplot(c.df, group_col), "eda_box",
                        "Spread of daily counts in each slot: the box holds the middle half of days.",
                        [("Why it matters", "Arrivals rise and fall within the lunch period, so one rate for the "
                          "whole period would be wrong. That is why λ is estimated per slot.")])
        elif view == "By weekday":
            chart_block(viz.weekday_profile(c.df, group_col), "eda_weekday",
                        "Average arrivals per slot for each weekday. Click a day in the legend to hide it.",
                        [("What to look for", "Lines that run close together mean the weekday has little effect. "
                          "Weekday effects are tested on the Peak forecast page.")])
        else:
            chart_block(viz.arrivals_heatmap(c.df), "eda_heatmap",
                        "Every recorded count: rows are days, columns are slots, darker means more arrivals.",
                        [("What to look for", "A dark vertical band is a slot busy on most days; paler rows are "
                          "quiet days such as exam weeks. Blank cells are recording gaps.")])
    with tabs[2]:
        r = c.report
        figures([(f"{r['rows_in']}", "records read"), (f"{r['missing_arrivals']}", "missing counts dropped"),
                 (f"{r['invalid_arrivals'] + r['invalid_dates'] + r['invalid_intervals']}", "invalid rows dropped"),
                 (f"{r['duplicates']}", "duplicates dropped"), (f"{len(c.df)}", "observations analysed")])
        with st.expander("How the data are collected, and the file format"):
            st.markdown(
                "- Arrivals are counted at the canteen entry (manual count or permitted video), following Tang et "
                "al. (2019a), in fixed 5-, 10- or 15-minute slots on m working days.\n"
                "- CSV columns: `date, time_interval, arrivals`, optional `day_type` (Regular / Exam), e.g. "
                "`2026-01-05, 12:30-12:40, 14, Regular`.\n"
                "- Missing counts are dropped, not filled in: imputing the mean would shrink the variance and bias "
                "the dispersion test.")
            if c.synthetic:
                st.info("Field data collection is in progress. Until then the analysis uses synthetic prototype "
                        "records from a seeded generator; they are not observed results.")
        preview = c.df.drop(columns=["hour_block"]).assign(date=c.df["date"].dt.date)
        st.dataframe(preview, hide_index=True, width="stretch", height=380)


def model_check() -> None:
    c = ctx()
    alpha, group_col, g = c.params["alpha"], c.params["group_col"], c.result.gof_summary
    ok = not g["excessive"]
    page_header("Analysis", "Is the Poisson model right for these arrivals?",
                (f"<b>Yes.</b> The Poisson model was rejected in {g['rejections']} of {g['tested']} slots, in line "
                 f"with the {g['expected_by_chance']:.1f} rejections expected by chance alone.") if ok else
                (f"<b>Questionable.</b> The Poisson model was rejected in {g['rejections']} of {g['tested']} slots, "
                 f"more than the {g['expected_by_chance']:.1f} expected by chance."),
                "Every probability in this project assumes Poisson counts, so the assumption is tested slot by slot "
                "(research gap G2) and checked by simulation.")
    if group_col == "hour_block":
        st.warning("Hourly slots pool short slots whose rates differ, which produces overdispersion even when every "
                   "short slot is Poisson (Kim & Whitt, 2014). Use individual slots for model checking.")
    tabs = st.tabs(["One slot in detail", "All slots", "Simulation"])
    with tabs[0]:
        period = slot_picker("pick_fit")
        observations = c.df.loc[c.df[group_col] == period, "arrivals"].to_numpy()
        gof = chi_square_gof(observations, alpha)
        disp = dispersion_test(observations, alpha)
        left, right = st.columns([2.3, 1], gap="large")
        with left:
            chart_block(viz.arrival_distribution(observations, gof.lambda_hat, ""), f"fit_hist_{period}",
                        "Bars: days on which each number of arrivals was seen. Line: what the Poisson model expects.",
                        [("The test", "H₀: counts follow Poisson(λ̂). The chi-square statistic adds up "
                          "(observed − expected)² / expected over categories, merged until every expected count "
                          f"is at least {MIN_EXPECTED_FREQUENCY:g} (Cochran, 1954).")])
        with right:
            figures([(f"{gof.chi_square:.2f}" if gof.valid else "n/a", "χ² statistic"),
                     (f"{gof.df}" if gof.valid else "n/a", "degrees of freedom"),
                     (f"{gof.p_value:.3f}" if gof.valid else "n/a", "p-value")], stacked=True)
        verdict("na" if not gof.valid else ("bad" if gof.reject_h0 else "ok"), html.escape(gof.message))
        verdict("bad" if disp["p_value"] < alpha else "ok",
                f"Mean vs variance: Var/Mean = {disp['index']:.2f}, D = {disp['statistic']:.1f} on {disp['df']} df, "
                f"p = {disp['p_value']:.3f} ({html.escape(disp['label'])}).")
        if not gof.table.empty:
            with st.expander("Categories used in the test"):
                cols = ["category", "observed", "expected"] + (["contribution"] if "contribution" in gof.table else [])
                st.dataframe(gof.table[cols], hide_index=True, width="stretch", column_config={
                    "category": "Category (x)", "observed": "Observed O",
                    "expected": st.column_config.NumberColumn("Expected E", format="%.2f"),
                    "contribution": st.column_config.NumberColumn("(O−E)²/E", format="%.3f")})
                for line in gof.merge_notes:
                    st.caption(line)
    with tabs[1]:
        figures([(f"{g['rejections']} of {g['tested']}", "slots where H₀ was rejected"),
                 (f"{g['expected_by_chance']:.1f}", "rejections expected by chance (α × m)"),
                 (f"{g['binom_p']:.3f}", "P(at least this many | all Poisson)")])
        if c.result.table["variance"].notna().any():
            chart_block(viz.mean_vs_variance_scatter(c.result.table), "fit_scatter",
                        "Each point is a slot. Poisson counts have variance equal to the mean, so points should lie "
                        "near the dashed line.",
                        [("Why two tests", "Equal mean and variance is necessary for a Poisson model but not proof "
                          "of one, which is why the chi-square test is also run."),
                         ("Multiple testing", "With many slots a few rejections happen by chance; the binomial "
                          "check above asks whether there are more than chance explains.")])
        with st.expander("Test results for every slot"):
            st.dataframe(c.result.gof_table, hide_index=True, width="stretch", column_config={
                "period": "Slot", "n": "Days m", "lambda_hat": st.column_config.NumberColumn("λ̂", format="%.3f"),
                "categories": "Categories k", "chi_square": st.column_config.NumberColumn("χ²", format="%.3f"),
                "df": "df", "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                "decision": "Decision", "valid": "Valid test"})
    with tabs[2]:
        _simulation()


@st.fragment
def _simulation() -> None:
    c = ctx()
    capacity = c.params["capacity"]
    c1, c2, c3 = st.columns([1.2, 1.4, 0.8], vertical_alignment="bottom")
    with c1:
        period = slot_picker("pick_sim")
    n_days = c2.select_slider("Simulated days", [100, 250, 500, 1000, 2500, 5000], value=1000, key="mc_days")
    seed = int(c3.number_input("Seed", 0, 99_999, 7, key="mc_seed"))
    row = _row(period)
    lam, p_model = float(row["lambda_hat"]), float(row["overload_prob"])
    _draws, running = simulate_overload_frequency(lam, capacity, n_days, seed)
    se = float(np.sqrt(p_model * (1 - p_model) / n_days))
    figures([(pct(p_model), "formula P(X > C)"), (pct(float(running[-1])), f"simulated, {n_days} days"),
             (f"±{100 * 2 * se:.2f} pts", "expected range (2 SE)")])
    chart_block(viz.monte_carlo_convergence(running, p_model), f"mc_{n_days}_{seed}_{period}",
                "Press Play: the share of simulated days with more than C arrivals settles on the formula value.",
                [("Why this matters", "A probability is a long-run frequency. The line wobbles at first and then "
                  "settles on the dashed value: the law of large numbers confirms the formula.")])


def overload() -> None:
    c = ctx()
    capacity, table, k = c.params["capacity"], c.result.table, c.result.kpis
    counts = k["risk_counts"]
    page_header("Analysis", "How likely is overcrowding in each slot?",
                f"Overcrowding is most likely at <b>{slot(k['max_overload_period'])}</b> "
                f"(P = {pct(k['max_overload_prob'], 1)}). {counts['HIGH']} slot(s) are high risk, "
                f"{counts['MEDIUM']} medium and {counts['LOW']} low.",
                f"Overcrowding means more than C = {capacity} arrivals in a slot: P(Xₜ > C) = P(Xₜ ≥ K) with "
                f"K = {capacity + 1} (research gaps G1 and G4).")
    tabs = st.tabs(["Risk by slot", "What if capacity changed?", "Capacity planning"])
    with tabs[0]:
        key = f"overflow_chart_{c.period}"
        chart_block(viz.overload_by_period(table, c.params["low"], c.params["high"]), key,
                    "P(X > C) for each slot, coloured by risk level. Click a bar to inspect that slot.",
                    [("Whiskers", "How the probability moves across the confidence interval for λ (gap G4)."),
                     ("Risk levels", f"Low below {100 * c.params['low']:.0f}%, High above "
                      f"{100 * c.params['high']:.0f}%, Medium in between. These are project decision rules, "
                      "adjustable under Advanced settings.")],
                    on_select=partial(ui.select_period_from_chart, key), selection_mode="points")
        ordered = table.sort_values("period")
        st.dataframe(pd.DataFrame({"Slot": ordered["period"].map(slot), "λ̂": ordered["lambda_hat"],
                                   "P(X > C)": 100 * ordered["overload_prob"],
                                   "Risk": ordered["risk"].map(ui.RISK_WORD), "Poisson fit": ordered["gof_decision"]}),
                     hide_index=True, width="stretch",
                     column_config={"λ̂": st.column_config.NumberColumn(format="%.2f"),
                                    "P(X > C)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                                                max_value=100)})
        with st.expander("Recommendations"):
            for rec in c.result.recommendations:
                st.markdown(f"- {rec}")
    with tabs[1]:
        period = slot_picker("pick_over")
        row = _row(period)
        lam = float(row["lambda_hat"])
        left, right = st.columns([2.3, 1], gap="large")
        with left:
            chart_block(viz.overload_curve(lam, capacity, slot(period)), f"overflow_curve_{period}",
                        "P(X > c) for this slot as the capacity c changes; the dashed line is the current capacity.")
        with right:
            figures([(pct(row["overload_prob"]), f"P(X > {capacity})"),
                     (f"{pct(row['overload_prob_low'], 1)} to {pct(row['overload_prob_high'], 1)}", "over the λ CI")],
                    stacked=True)
            html_block(f"<p>Risk level {pill(row['risk'])}</p>")
        with st.expander(f"Worked calculation for {slot(period)}"):
            breakdown = overload_breakdown(lam, capacity)
            st.dataframe(breakdown, hide_index=True, height=240, width="stretch", column_config={
                "P(X = x)": st.column_config.NumberColumn(format="%.6f"),
                "P(X ≤ x)": st.column_config.NumberColumn(format="%.6f")})
            st.markdown(f"P(X ≤ {capacity}) = {breakdown['P(X ≤ x)'].iloc[-1]:.6f}, so P(X > {capacity}) = "
                        f"**{overload_probability_manual(lam, capacity):.6f}** by the textbook sum; SciPy's "
                        f"`poisson.sf({capacity}, {lam:.3f})` gives **{overload_probability(lam, capacity):.6f}**.")
        if st.toggle("Show every rate and capacity in 3D", key="over_3d"):
            chart_block(viz.overload_surface_3d(table, capacity), "overflow_surface",
                        "P(X > C) for every rate λ and capacity C; dots are the slots at the current capacity. "
                        "Drag to rotate.")
    with tabs[2]:
        target = st.slider("Keep the chance of overcrowding below (%)", 1, 50, int(100 * c.params["low"]),
                           key="plan_target") / 100
        ordered = table.sort_values("period")
        needed = [required_capacity(float(lam), target) for lam in ordered["lambda_hat"]]
        short = [p for p, n in zip(ordered["period"], needed) if n > capacity]
        verdict("na" if short else "ok",
                (f"{len(short)} slot(s) need more than the current {capacity}: "
                 + ", ".join(slot(p) for p in short) + f". The peak needs {max(needed)}.")
                if short else f"The current capacity of {capacity} meets the target in every slot.")
        chart_block(viz.required_capacity_chart(ordered["period"].tolist(), needed, capacity, target), "capacity_plan",
                    f"Smallest capacity that keeps P(overcrowding) below {100 * target:.0f}% in each slot; red bars "
                    "exceed the current capacity.")


def forecast() -> None:
    c = ctx()
    model, disp, effects = _forecast_inputs()
    capacity, var_w = c.params["capacity"], disp["var_w"]
    _title("Analysis", "Can we predict the peak time?")
    weekday, day_type = _day_pickers(model, "fc")
    mu = model.predict(weekday, day_type)
    p_over = pd.Series(fc.overload_probability_mixed(mu.to_numpy(), capacity, var_w), index=mu.index)
    pk = _peak(mu, capacity, var_w)
    windows = contiguous_windows(sorted(pk["top_two"]), [True, True])
    top = window_label(windows[0]) if len(windows) == 1 else " or ".join(slot(p) for p in sorted(pk["top_two"]))
    html_block(f'<div class="answer-box">On a {day_type.lower()} {weekday} the peak is most likely at '
               f'<b>{slot(pk["most_likely_peak"])}</b> ({pct(pk["p_most_likely"], 0)} chance), and falls in {top} '
               f'with probability <b>{pct(pk["p_top_two"], 0)}</b>. Chance that at least one slot exceeds C = {capacity}: '
               f'<b>{pct(pk["p_any_overload"], 0)}</b>.</div>')
    tabs = st.tabs(["Peak and risk", "Update during the day", "How the forecast is made"])
    with tabs[0]:
        chart_block(viz.peak_probability_bars(pk["p_peak"]), "fc_peak",
                    f"Share of {pk['n_sims']:,} simulated lunch periods in which each slot had the most arrivals.",
                    [("Why the peak is uncertain", "Neighbouring slots have similar rates, so on any one day either "
                      "may turn out busiest (research gap G8)."),
                     ("Method", "Each simulated day draws a busyness factor W ~ Gamma(mean 1, variance v̂), then "
                      "Xₜ ~ Poisson(W·λₜ) for every slot (Modules 1 and 2).")])
        risk = [classify_risk(p, c.params["low"], c.params["high"]) for p in p_over]
        st.dataframe(pd.DataFrame({"Slot": [slot(p) for p in mu.index], "Predicted λ": mu.to_numpy(),
                                   "P(X > C)": 100 * p_over.to_numpy(), "Risk": [ui.RISK_WORD[r] for r in risk],
                                   "P(peak)": 100 * pk["p_peak"].to_numpy()}),
                     hide_index=True, width="stretch",
                     column_config={"Predicted λ": st.column_config.NumberColumn(format="%.2f"),
                                    "P(X > C)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                                                max_value=100),
                                    "P(peak)": st.column_config.NumberColumn(format="%.1f%%")})
        st.markdown("#### Risk by weekday and time")
        grid = pd.DataFrame({w: fc.overload_probability_mixed(model.predict(w, day_type).to_numpy(), capacity, var_w)
                             for w in model.weekdays}, index=model.periods).T
        chart_block(viz.forecast_surface_3d(grid, weekday), "fc_surface",
                    f"Predicted P(X > {capacity}) for every weekday and slot; the red line is the selected day. "
                    "Drag to rotate.",
                    [("Why it is a forecast", "Each value comes from the regression rate for that weekday and day "
                      "type, not from an average of past days, and includes day-to-day variation (G7, G10).")])
    with tabs[1]:
        _same_day_update(mu, capacity, var_w)
    with tabs[2]:
        _regression_view(model, effects, c.params["alpha"])


@st.fragment
def _same_day_update(mu: pd.Series, capacity: int, var_w: float) -> None:
    periods = list(mu.index)
    if len(periods) < 2:
        st.info("Updating during the day needs at least two time slots per day.")
        return
    k = st.slider("Slots already counted today", 1, len(periods) - 1, min(2, len(periods) - 1), key="fc_known")
    st.caption("Enter today's counts so far; the forecast for each slot is shown in brackets.")
    cols = st.columns(min(k, 6))
    observed = {p: cols[i % len(cols)].number_input(f"{p[:5]} ({mu[p]:.0f})", 0, 1000, int(round(mu[p] * 1.15)),
                                                    key=f"fc_obs_{p}")
                for i, p in enumerate(periods[:k])}
    table = fc.same_day_update(mu, pd.Series(observed), capacity, var_w)
    ratio = sum(observed.values()) / mu.iloc[:k].sum()
    figures([(f"{100 * ratio:.0f}%", "arrivals so far, % of forecast"),
             (f"{table['w_mean'].iloc[0]:.3f}", "today's busyness factor (posterior mean)"),
             (f"± {table['w_sd'].iloc[0]:.3f}", "its standard deviation")])
    chart_block(viz.update_before_after(table), "fc_update_bars",
                "Chance of overcrowding in the remaining slots, before and after today's counts.",
                [("Bayes theorem", "Prior W ~ Gamma(a, a) with a = 1/v̂. After counts x with forecasts λ, the "
                  "posterior is Gamma(a + Σx, a + Σλ); each later slot follows a negative binomial (gap G9)."),
                 ("Why the change can be small", "When days differ little (small v̂) the prior is strong and a "
                  "busy morning moves the forecast only a little; with more day-to-day variation it moves further.")])
    if st.toggle("Show every possible morning in 3D", key="fc_update_3d"):
        chart_block(viz.update_surface_3d(mu, capacity, var_w, k), f"fc_update_surface_{k}",
                    "Height: updated P(X > C) of each remaining slot; depth: how busy the counted slots were "
                    "relative to the forecast. Dashed line: a day exactly as forecast.")


def _regression_view(model, effects: dict, alpha: float) -> None:
    figures([(f"{model.r_squared:.3f}", "R² (log scale)"), (f"{model.adj_r_squared:.3f}", "adjusted R²"),
             (f"{model.n}", "observations"), (f"{model.dof}", "residual df")])
    chart_block(viz.coefficient_chart(model.coefficients), "fc_coef",
                "Each effect as a multiplier on expected arrivals, with a 95% interval: teal raises arrivals, blue "
                f"lowers them, grey is not significant at α = {alpha}.",
                [("Model", "log(x + 0.5) = b₀ + slot + weekday + day type + e, fitted by least squares (multiple "
                  "linear regression, Module 3); each coefficient has a t-test (Module 5). Means are "
                  f"back-transformed with the smearing factor {model.smear:.3f} (Duan, 1983).")])
    wk, dt = effects.get("weekday"), effects.get("day_type")
    if wk:
        sig = wk["anova_p"] < alpha
        verdict("ok" if sig else "na",
                f"<b>Weekday</b> (regular days, daily totals): ANOVA F = {wk['anova_f']:.2f}, p = {_p(wk['anova_p'])}; "
                f"Kruskal-Wallis H = {wk['kruskal_h']:.2f}, p = {_p(wk['kruskal_p'])}. "
                + ("Weekdays differ." if sig else "No significant difference in daily totals."))
    if dt:
        sig = dt["t_p"] < alpha
        verdict("ok" if sig else "na",
                f"<b>Exam vs regular days</b>: {dt['special_mean']:.1f} vs {dt['regular_mean']:.1f} arrivals per lunch "
                f"({dt['n_special']} vs {dt['n_regular']} days); Welch t = {dt['t']:.2f}, p = {_p(dt['t_p'])}; "
                f"Mann-Whitney U = {dt['u']:.0f}, p = {_p(dt['u_p'])}. "
                + ("Exam days differ." if sig else "No significant difference."))
    elif len(model.day_types) == 1:
        st.caption("All days are of one type. Add a day_type column (e.g. Regular / Exam) to field data to estimate "
                   "special-day effects.")
    with st.expander("Coefficient table"):
        coef = model.coefficients.assign(p_value=model.coefficients["p_value"].map(_p))
        st.dataframe(coef.rename(columns={"term": "Term", "estimate": "Estimate", "std_error": "Std. error",
                                          "t": "t", "p_value": "p-value", "multiplier": "Multiplier"}),
                     hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.4f")
                                    for k in ("Estimate", "Std. error", "t", "Multiplier")})


def forecast_check() -> None:
    c = ctx()
    model, disp, _ = _forecast_inputs()
    capacity, alpha = c.params["capacity"], c.params["alpha"]
    n_days, n_slots = c.df["date"].nunique(), c.df["time_interval"].nunique()
    _title("Analysis", "Can we trust the forecast?")
    if n_days < 16:
        st.warning("At least 16 days are needed to test the forecasts. Widen the filters under Advanced settings.")
        st.stop()
    with st.expander("Test settings", icon=":material/tune:"):
        c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
        train = c1.slider("Days before the first forecast", 10, min(60, n_days - 5), min(30, n_days - 5),
                          key="bt_train")
        window = c2.radio("Learn from", ["All earlier days", f"Last {train} days only"], key="bt_window")
        upd = (c3.slider("Update after slot", 1, n_slots - 1, min(2, n_slots - 1), key="bt_update")
               if n_slots > 2 else 1)
    try:
        bt, s = _backtest(c.df, capacity, train, upd, window.startswith("All"), alpha)
    except ValueError as err:
        st.warning(str(err))
        st.stop()
    paired = s["tests"].iloc[2]
    significant = paired["p_value"] < alpha
    html_block(
        f'<div class="answer-box">Each of {s["n_days"]} past days was forecast using only the days before it. The '
        f'95% ranges covered <b>{pct(s["coverage"], 0)}</b> of actual counts and the busiest slot was named '
        f'correctly <b>{pct(s["hit_rate_model"], 0)}</b> of the time (chance: {pct(s["hit_chance"], 0)}). Against '
        f'the simple per-slot average the error is {s["mae_model"]:.2f} vs {s["mae_base"]:.2f} arrivals per slot, '
        + ('a <b>significant</b> improvement.' if significant else '<b>not a significant</b> difference on this '
           'data.') + '</div>')
    figures([(f"{s['brier_model']:.4f}", "Brier score, model"), (f"{s['brier_base']:.4f}", "Brier score, average"),
             (f"{100 * s['brier_skill']:+.1f}%", "Brier skill vs average"),
             (f"{s['pearson_r']:.3f}", "correlation, predicted vs actual")])
    tabs = st.tabs(["Test results", "Calibration and errors", "Variation between days"])
    with tabs[0]:
        t = s["tests"]
        st.dataframe(pd.DataFrame({
            "Check": t["check"], "Result": t["result"], "p-value": t["p_value"].map(_p),
            "Supports the model": t["supports_model"].map({True: "✅ Yes", False: "❌ No", None: "n/a"}),
            "Module": t["module"]}),
            hide_index=True, width="stretch",
            column_config={"Check": st.column_config.TextColumn(width="large"),
                           "Module": st.column_config.TextColumn(width="small")})
        with st.expander("Test methods and statistics"):
            st.dataframe(pd.DataFrame({"Check": t["check"], "Method": t["method"], "Statistic": t["statistic"]}),
                         hide_index=True, width="stretch")
            st.caption(f"α = {alpha}. Coverage and the PIT tests support the model when H₀ is not rejected; the "
                       "hit-rate and model-vs-average tests support it when H₀ is rejected.")
        if "brier_updated" in s:
            gain = 1 - s["brier_updated"] / s["brier_not_updated"]
            verdict("ok" if gain > 0 else "na",
                    f"<b>Updating during the day</b> (after slot {upd}) makes the later forecasts "
                    f"{100 * abs(gain):.1f}% {'better' if gain > 0 else 'worse'}: Brier score "
                    f"{s['brier_updated']:.4f} with the update, {s['brier_not_updated']:.4f} without.")
        if not significant:
            st.markdown("With mild weekday and exam effects in the prototype data, the per-slot average is hard to "
                        "beat on accuracy (Ibrahim et al., 2016). The forecast adds what the average cannot: day-type "
                        "forecasts, the probability of each slot being the peak, calibrated ranges and updating "
                        "during the day.")
    with tabs[1]:
        left, right = st.columns(2, gap="large")
        with left:
            chart_block(viz.reliability_chart(s["reliability_model"], s["reliability_base"]), "bt_rel",
                        "Forecast probability against how often overcrowding happened; on the diagonal = calibrated.")
        with right:
            chart_block(viz.pit_histogram(s["pit_counts"]), "bt_pit",
                        "Where each actual count fell in its forecast distribution; flat = calibrated.",
                        [("Tests", "Kolmogorov–Smirnov and chi-square tests of uniformity (Module 5). A pile-up "
                          "near 1 means actual counts were often higher than forecast.")])
        chart_block(viz.daily_error_comparison(bt.days), "bt_daily",
                    "Each past day's average error: below the diagonal the model was more accurate than the average.")
        st.dataframe(s["mae_by_group"].rename(columns={"group": "Days", "days": "n", "mae_model": "Error, model",
                                                       "mae_base": "Error, average"}),
                     hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.2f")
                                    for k in ("Error, model", "Error, average")})
        if st.toggle("Show forecasts against actual counts in 3D", key="bt_3d"):
            chart_block(viz.backtest_surface_3d(bt.slots), "bt_surface",
                        "Surface: what arrived on each past day. Orange points: the forecasts made the day before.")
    with tabs[2]:
        _overdispersion_view(disp, capacity, alpha)
        st.markdown("#### Do busy periods carry over?")
        corr = fc.consecutive_slot_correlation(c.df, model)
        runs = fc.daily_residual_runs(c.df, model)
        left, right = st.columns(2, gap="large")
        with left:
            st.dataframe(corr.assign(pair=corr["pair"].map(slot), p_value=corr["p_value"].map(_p))
                         .rename(columns={"pair": "Consecutive slots", "rho": "Spearman ρ", "p_value": "p-value"}),
                         hide_index=True, width="stretch",
                         column_config={"Spearman ρ": st.column_config.NumberColumn(format="%.3f")})
            st.caption("Rank correlation of observed ÷ predicted between neighbouring slots (Module 3). Positive ρ "
                       "is what makes updating during the day work.")
        with right:
            verdict("na" if runs["p_value"] < alpha else "ok",
                    f"<b>Busy days after busy days?</b> {runs['runs']} runs against {runs['expected']:.1f} expected "
                    f"({runs['n_pos']} days above forecast, {runs['n_neg']} below); z = {runs['z']:.2f}, "
                    f"p = {_p(runs['p_value'])}. "
                    + ("Days are not independent." if runs["p_value"] < alpha else
                       "No evidence that busy days cluster."))
            st.caption("Wald–Wolfowitz run test on the sign of each day's residual (Module 5).")


def _overdispersion_view(disp: dict, capacity: int, alpha: float) -> None:
    table = ctx().result.table
    sig = disp["p_value"] < alpha
    verdict("na" if sig else "ok",
            f"<b>Do whole days vary more than Poisson allows?</b> Pearson χ² = {disp['pearson_chi2']:.1f} on "
            f"{disp['dof']} df, p = {_p(disp['p_value'])}. "
            + (f"Yes: day-to-day variation v̂ = {disp['var_w']:.4f} (CV {100 * disp['cv_w']:.1f}%), so the "
               "Poisson–Gamma probabilities are used." if sig else
               f"No significant extra variation (v̂ = {disp['var_w']:.4f}); the correction below is small."))
    p_mixed = fc.overload_probability_mixed(table["lambda_hat"].to_numpy(), capacity, disp["var_w"])
    chart_block(viz.poisson_vs_mixed_bars(table["period"].tolist(), table["overload_prob"], p_mixed), "bt_mixed",
                "P(X > C) per slot under the plain Poisson model and with day-to-day variation (gap G10).",
                [("Model", "By the theorem of total probability over the day's busyness factor W ~ Gamma(1/v, 1/v), "
                  "X is negative binomial with mean λ and variance λ + vλ² (Modules 1 and 2). v̂ = (χ² − df) / Σλ "
                  "(Cameron & Trivedi, 1990).")])


# ===========================================================================
# REPORT
# ===========================================================================
def project() -> None:
    c = ctx()
    k, q, p = c.result.kpis, c.result.quiet, c.params
    team = "".join(f"<b>{name}</b> ({reg})<br>" for name, reg in TEAM)
    html_block(
        '<div class="cover"><h1>Statistical Estimation of Canteen Arrival Overload Probability Using Poisson '
        'Distribution</h1><p class="sub">CanteenFlow: predicting when the canteen gets crowded</p>'
        '<div class="meta">'
        '<div><span>Course</span><b>PAMAT501, Probability and Statistics</b></div>'
        f'<div><span>Presented by</span>{team}</div>'
        '<div><span>Faculty guide</span><b>Dr. Manimaran</b></div>'
        '<div><span>Base paper</span><b>Tang et al. (2019), Physica A, 521, 29–40</b></div>'
        '</div></div>')
    tabs = st.tabs(["Summary", "Literature", "Research gaps"])
    with tabs[0]:
        st.markdown(
            f"University canteens see sharp changes in arrival intensity during lunch. For each {c.minutes}-minute "
            f"slot this project estimates the probability that arrivals exceed the counters' capacity, using "
            f"{k['n_days']} working days ({k['total_observations']:,} slot observations). The arrival rate λₜ is "
            f"estimated per slot, the Poisson assumption is tested, and P(Xₜ > C) is computed for C = "
            f"{p['capacity']}: the riskiest slot is **{slot(k['max_overload_period'])}** "
            f"(P = {pct(k['max_overload_prob'], 1)}) and the least crowded **{slot(q['best'])}**. The second review "
            "adds **prediction**: day-specific rates by regression, the probability that each slot is the day's "
            "peak, updating from the morning's counts by Bayes theorem, and a test of every forecast on unseen "
            "days. All methods come from the PAMAT501 syllabus.")
        if c.synthetic:
            st.caption("The bundled records are synthetic prototype data; field counts can be uploaded under Data.")
        html_block('<div class="flow"><span>Observed arrivals</span><i>→</i><span>Estimate λₜ</span><i>→</i>'
                   '<span>Test Poisson fit</span><i>→</i><span>P(Xₜ > C)</span><i>→</i><span>Predict the peak</span>'
                   '<i>→</i><span>Validate on new days</span></div>')
        st.markdown("#### Objectives")
        objectives = [("Collect and organise arrival data", "arrivals"),
                      ("Estimate time-specific rates and test the Poisson model", "model"),
                      ("Calculate the overload probability and risk levels", "overload"),
                      ("Identify the least-crowded time", "plan"),
                      ("Predict the peak time and update it during the day", "forecast"),
                      ("Validate the forecasts on unseen days", "check")]
        cols = st.columns(2, gap="medium")
        for i, (title, page_key) in enumerate(objectives):
            with cols[i % 2], st.container(border=True):
                st.markdown(f"**{i + 1}. {title}**")
                _link(page_key, "Open")
        st.markdown("#### Syllabus map")
        _syllabus_table()
    with tabs[1]:
        _literature()
    with tabs[2]:
        _gaps()


def _syllabus_table() -> None:
    short = {"M1": "Probability & random variables", "M2": "Distributions", "M3": "Correlation & regression",
             "M4": "Large-sample tests", "M5": "Small-sample & non-parametric tests"}
    counts = {m: sum(m in row[2] for row in SYLLABUS_MAP) for m in SYLLABUS_MODULES}
    chosen = st.pills("Modules", list(SYLLABUS_MODULES), selection_mode="multi", default=list(SYLLABUS_MODULES),
                      format_func=lambda m: f"{m} {short[m]} ({counts[m]})", key="syl_filter",
                      label_visibility="collapsed") or list(SYLLABUS_MODULES)
    rows = [{"Component": comp, "Syllabus topics": topics, "Modules": ", ".join(mods), "Page": where, "Gap": gap}
            for comp, topics, mods, where, gap in SYLLABUS_MAP if set(mods) & set(chosen)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(36 * len(rows) + 40, 560),
                 column_config={"Component": st.column_config.TextColumn(width="medium"),
                                "Syllabus topics": st.column_config.TextColumn(width="medium"),
                                "Modules": st.column_config.TextColumn(width="small"),
                                "Page": st.column_config.TextColumn(width="small"),
                                "Gap": st.column_config.TextColumn(width="small")})
    st.caption("Every statistical component and its PAMAT501 module (M1–M5). Methods from the forecasting papers "
               "that lie outside the syllabus (factor models, MCMC, integer-GARCH, machine learning) are cited as "
               "context only.")


def _literature() -> None:
    st.markdown(f"{len(PAPERS)} studies: the base paper, nine from the first review and eight read for the second "
                "review on forecasting arrivals. **Select a row** to read a study's method, findings and what it "
                "leaves open.")
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


def _gaps() -> None:
    st.markdown("Eleven gaps: G1 from the first review, G2–G6 from the extended review, G7–G11 from the "
                "second-review reading on forecasting. Each is a focus and application gap, not a claim that "
                "Poisson modelling of arrivals is new.")
    labels = [g["id"] for g in GAPS]
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
    with st.expander("All gaps at a glance"):
        st.markdown("| Gap | In one line |\n|---|---|\n" + "\n".join(
            f'| **{x["id"]}** {x["title"]} | {x["summary"]} |' for x in GAPS))


FORMULA_TOPICS = ["Poisson model", "Estimating λ", "Overload probability", "Dispersion test", "Goodness of fit",
                  "Comparing two slots", "Regression forecast", "Day-to-day variation", "Same-day update",
                  "Scoring forecasts"]


def methodology() -> None:
    c = ctx()
    page_header("Report", "How is the analysis done?",
                "Nine steps: <b>observe → clean → group → estimate → test fit → compute → present</b> from the first "
                "review, then <b>forecast → validate</b> from the second. The Poisson model is checked against the "
                "data, not assumed.")
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
    tabs = st.tabs(["Steps", "Formulas", "Poisson explorer"])
    with tabs[0]:
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
    with tabs[1]:
        topic = st.selectbox("Show the formulas for", FORMULA_TOPICS, key="formula_topic")
        _formulas(c, topic)
    with tabs[2]:
        _poisson_explorer(float(_row(c.period)["lambda_hat"]), c.params["capacity"], c.period)


def _formulas(c, topic: str) -> None:
    if topic == FORMULA_TOPICS[0]:
        st.markdown("Xₜ is the number of students arriving during slot t. The model assumes")
        st.latex(r"X_t \sim \mathrm{Poisson}(\lambda_t),\qquad P(X_t = x) = \frac{e^{-\lambda_t}\lambda_t^{x}}{x!},"
                 r"\quad x = 0, 1, 2, \dots")
        st.latex(r"E[X_t] = \mathrm{Var}(X_t) = \lambda_t")
        st.markdown("λₜ is estimated separately for each slot because arrival intensity changes across the peak.")
    if topic == FORMULA_TOPICS[1]:
        st.markdown("For slot t observed on m days, the maximum-likelihood estimate is the sample mean:")
        st.latex(r"\ln L(\lambda) = -m\lambda + \ln\lambda\sum x_i - \sum \ln x_i!\ \Rightarrow\ "
                 r"\hat\lambda_t = \frac{1}{m}\sum_{i=1}^{m} x_i")
        st.markdown("Exact (Garwood, 1936) confidence interval, using S = Σxᵢ ~ Poisson(mλ):")
        st.latex(r"\left[\frac{\chi^2_{\alpha/2}(2S)}{2m},\ \frac{\chi^2_{1-\alpha/2}(2S+2)}{2m}\right]")
    if topic == FORMULA_TOPICS[2]:
        st.markdown("Overload means more than C students arrive in a slot. With K = C + 1 this is the "
                    "first-review notation P(Xₜ ≥ K):")
        st.latex(r"P(X_t > C) = P(X_t \ge K) = 1 - \sum_{x=0}^{C} \frac{e^{-\hat\lambda_t}\hat\lambda_t^{x}}{x!}")
        st.markdown(f"Risk label: Low if P < {100 * c.params['low']:.0f}%, High if P > {100 * c.params['high']:.0f}%, "
                    "Medium otherwise (thresholds set in the sidebar).")
    if topic == FORMULA_TOPICS[3]:
        st.markdown("Index-of-dispersion test of Var = Mean, two-sided:")
        st.latex(r"D = \frac{(m-1)s^2}{\bar x} \sim \chi^2_{m-1}\ \text{under } H_0")
        st.markdown("Var/Mean > 1 indicates overdispersion (for example group arrivals); < 1 underdispersion.")
    if topic == FORMULA_TOPICS[4]:
        st.markdown("H₀: counts follow Poisson(λ̂). H₁: they do not.")
        st.latex(r"\chi^2 = \sum_{i=1}^{k}\frac{(O_i - E_i)^2}{E_i},\qquad df = k - 2")
        st.markdown(f"Adjacent counts are merged until every expected frequency is at least "
                    f"{MIN_EXPECTED_FREQUENCY:g} (Cochran, 1954). One df is lost because probabilities sum to 1 and "
                    "one because λ is estimated.")
    if topic == FORMULA_TOPICS[5]:
        st.markdown("To check that the least-crowded slot is genuinely quieter than the runner-up "
                    "(Przyborowski & Wileński, 1940), condition on the combined total:")
        st.latex(r"S_1 \mid (S_1 + S_2) \sim \mathrm{Binomial}\!\left(S_1 + S_2,\ \frac{m_1}{m_1 + m_2}\right)"
                 r"\ \text{under } H_0: \lambda_1 = \lambda_2")
        st.markdown("The one-sided p-value is P(Binomial ≤ S₁).")
    if topic == FORMULA_TOPICS[6]:
        st.markdown("Multiple linear regression (Module 3) on the log of the counts, with dummy variables for "
                    "slot t, weekday d and day type e (reference levels dropped):")
        st.latex(r"\ln(x_{dt} + 0.5) = \beta_0 + \alpha_t + \gamma_{d} + \delta_{e} + \varepsilon_{dt},\qquad "
                 r"\hat\beta = (X^\top X)^{-1}X^\top y")
        st.latex(r"\hat\lambda_{dt} = \exp(x_{dt}^\top\hat\beta)\cdot\overline{e^{\hat\varepsilon}} - 0.5,\qquad "
                 r"t_j = \hat\beta_j / \mathrm{SE}(\hat\beta_j) \sim t_{n-p}")
        st.markdown("Weekday effects are also tested on daily totals with one-way ANOVA and Kruskal-Wallis, exam "
                    "days with Welch's t-test and Mann-Whitney U (Module 5).")
    if topic == FORMULA_TOPICS[7]:
        st.markdown("Each day has a busyness factor W with mean 1 and variance v. By the theorem of total "
                    "probability (Module 1) with W ~ Gamma (Module 2):")
        st.latex(r"X \mid W \sim \mathrm{Poisson}(W\lambda),\ W \sim \mathrm{Gamma}(1/v, 1/v)\ \Rightarrow\ "
                 r"E[X] = \lambda,\ \mathrm{Var}(X) = \lambda + v\lambda^2")
        st.latex(r"\hat v = \frac{\chi^2_P - (n - p)}{\sum \hat\lambda},\qquad "
                 r"\chi^2_P = \sum \frac{(x - \hat\lambda)^2}{\hat\lambda} \sim \chi^2_{n-p}\ \text{if Poisson holds}")
    if topic == FORMULA_TOPICS[8]:
        st.markdown("Bayes theorem (Module 1). Prior W ~ Gamma(a, a), a = 1/v. After today's first k slots:")
        st.latex(r"W \mid x_1,\dots,x_k \sim \mathrm{Gamma}\Big(a + \sum_{t\le k} x_t,\ a + \sum_{t\le k}\lambda_t\Big)")
        st.markdown("Each later slot then has a negative binomial predictive distribution, which gives the updated "
                    "P(Xₜ > C). P(slot t is the peak) is estimated by simulating whole days.")
    if topic == FORMULA_TOPICS[9]:
        st.markdown("Overload in a slot is a yes/no event o ∈ {0, 1} with forecast probability p (Brier, 1950; "
                    "McNulty, 1988):")
        st.latex(r"B = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2,\qquad \mathrm{BSS} = 1 - \frac{B_{\text{model}}}"
                 r"{B_{\text{per-slot average}}}")
        st.latex(r"z = \frac{\hat p - p_0}{\sqrt{p_0(1 - p_0)/n}}\quad\text{(coverage, hit rate; Module 4)}")


@st.fragment
def _poisson_explorer(lam_slot: float, capacity: int, period: str) -> None:
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
    figures([(pct(p_over), f"P(X > {cap})"), (pct(1 - p_over), f"P(X ≤ {cap})"),
             (f"{lam:.1f}", "mean = variance"), (f"{np.sqrt(lam):.2f}", "standard deviation √λ")])
    chart_block(viz.poisson_explorer(lam, cap), "explorer_chart",
                "Each bar is P(X = x); red bars are outcomes above capacity, and their total is the overload "
                "probability.",
                [("Try this", "Raise λ: the distribution shifts right and spreads by √λ. Raise C near the mean: the "
                  "red tail shrinks quickly, so a small capacity increase at the peak buys a large drop in risk.")])


def references() -> None:
    c = ctx()
    page_header("Report", "Limitations and references",
                lead="What the model leaves out, how the work can be extended, the full reference list and the "
                     "data behind every result.")
    tabs = st.tabs(["Limitations and future scope", "References", "Downloads"])
    with tabs[0]:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Limitations")
            st.markdown(
                "- **Prototype data.** Field collection is in progress; the bundled records are synthetic.\n"
                "- **Arrival risk, not queue length.** Queue length also depends on service rate, number of "
                "counters and queue discipline.\n"
                "- **Constant rate within a slot.** The Poisson model assumes λ does not change inside a slot.\n"
                "- **Independence.** Students arriving in groups cause overdispersion.\n"
                "- **Threshold choice.** Capacity C (or K) must be justified for the canteen and slot length.\n"
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
    with tabs[1]:
        st.markdown("#### Literature")
        for i, p in enumerate(sorted(PAPERS, key=lambda x: x["reference"]), start=1):
            doi = f" https://doi.org/{p['doi']}" if p["doi"] else ""
            st.markdown(f"{i}. {p['reference']}{doi}")
        st.markdown("#### Statistical methods")
        for i, ref in enumerate(METHOD_REFERENCES, start=len(PAPERS) + 1):
            st.markdown(f"{i}. {ref}")
    with tabs[2]:
        preview = c.df.drop(columns=["hour_block"]).assign(date=c.df["date"].dt.date)
        c1, c2 = st.columns(2)
        c1.download_button("Download analysed records (CSV)", preview.to_csv(index=False),
                           "canteen_arrivals_filtered.csv", "text/csv", key="dl_records", width="stretch")
        c2.download_button("Download slot results (CSV)", c.result.table.to_csv(index=False),
                           "canteenflow_results.csv", "text/csv", key="dl_results", width="stretch")
