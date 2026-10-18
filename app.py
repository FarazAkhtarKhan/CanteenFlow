"""
app.py — CanteenFlow Streamlit dashboard
=========================================
Statistical Estimation of Canteen Arrival Overload Probability Using Poisson Distribution.

Run with:   streamlit run app.py

This file is the entry point: it builds the sidebar settings, loads and cleans
the data, runs the statistical pipeline once, and hands the results to the
selected page (report_pages.py) through ui.CTX. All statistics live in
poisson_model.py, goodness_of_fit.py and analysis.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import report_pages as pages
import ui
from analysis import DEMAND_METHODS, GRANULARITIES, filter_dataset, run_pipeline
from data_generator import (DEFAULT_N_DAYS, DEFAULT_SEED, SUPPORTED_INTERVALS, default_capacity_for,
                            generate_dataset, save_dataset)
from utils import (DEFAULT_ALPHA, DEFAULT_DATA_PATH, DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD, clean_dataset,
                   dataset_default_capacity, load_dataset, slot_minutes, validate_thresholds)

st.set_page_config(page_title="CanteenFlow: Canteen Overload Probability", page_icon="🍽️", layout="wide")

SOURCES = ["Bundled lunch records", "Simulate new records", "Upload field data (CSV)"]


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bundled_dataset() -> pd.DataFrame:
    if not DEFAULT_DATA_PATH.exists():  # first run on a fresh copy: create the dataset
        save_dataset(generate_dataset())
    return load_dataset(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def generate_cached(n_days: int, seed: int, minutes: int) -> pd.DataFrame:
    return generate_dataset(n_days=n_days, seed=seed, interval_minutes=minutes,
                            capacity=default_capacity_for(minutes))


@st.cache_data(show_spinner=False)
def clean_cached(raw: pd.DataFrame):
    return clean_dataset(raw)


@st.cache_data(show_spinner=False)
def pipeline_cached(df, capacity, alpha, low, high, group_col, method):
    return run_pipeline(df, capacity, alpha, low, high, group_col, method)


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
def sidebar_dataset() -> tuple[pd.DataFrame, str, bool]:
    """Returns (raw data, description, synthetic?)."""
    with st.sidebar.expander("Dataset"):
        source = st.radio("Data source", SOURCES, key="sidebar_data_source", label_visibility="collapsed")
        if source == SOURCES[1]:
            minutes = st.select_slider("Slot length (minutes)", list(SUPPORTED_INTERVALS), value=10,
                                       key="sidebar_minutes")
            n_days = st.number_input("Working days", 30, 365, DEFAULT_N_DAYS, step=5, key="sidebar_n_days")
            seed = st.number_input("Random seed", 0, 10_000_000, DEFAULT_SEED, step=1, key="sidebar_seed")
            return (generate_cached(int(n_days), int(seed), int(minutes)),
                    f"synthetic records, {n_days} days, {minutes}-minute slots, seed {seed}", True)
        if source == SOURCES[2]:
            uploaded = st.file_uploader("CSV with columns date, time_interval, arrivals", type="csv",
                                        key="sidebar_upload")
            if uploaded is None:
                st.info("Upload a CSV in the sidebar (Dataset) to analyse field data.")
                st.stop()
            return load_dataset(uploaded), f"field data from {uploaded.name}", False
    bundled = load_bundled_dataset()
    return (bundled, f"bundled synthetic records, {bundled['date'].nunique()} days (one semester with two exam "
                     "weeks), 15-minute slots", True)


def sidebar_parameters(clean_df: pd.DataFrame, minutes: int) -> dict:
    st.sidebar.markdown("**Model parameters**")
    capacity = st.sidebar.number_input(
        f"Capacity C per {minutes}-minute slot", min_value=1, max_value=1000,
        value=dataset_default_capacity(clean_df), step=1, key=f"sidebar_capacity_{minutes}",
        help="Overload means more than C arrivals in a slot, i.e. X ≥ K with threshold K = C + 1.")
    alpha = st.sidebar.select_slider("Significance level α", options=[0.01, 0.025, 0.05, 0.10],
                                     value=DEFAULT_ALPHA, key="sidebar_alpha")
    low, high = st.sidebar.slider("Risk thresholds for P(overload), %", 1, 99,
                                  (int(100 * DEFAULT_LOW_THRESHOLD), int(100 * DEFAULT_HIGH_THRESHOLD)),
                                  key="sidebar_thresholds",
                                  help="Low below the first value, High above the second, Medium in between.")
    try:
        low_p, high_p = validate_thresholds(low / 100, high / 100)
    except ValueError as err:
        st.sidebar.error(str(err))
        st.error("Move the two risk thresholds apart: the low threshold must be smaller than the high one.")
        st.stop()
    with st.sidebar.expander("Analysis options"):
        granularity = st.radio("Slot grouping", list(GRANULARITIES), key="sidebar_granularity",
                               format_func=lambda g: "Individual slots" if g.startswith("15") else "Pooled by hour",
                               help="Pooling by hour shows why short slots are needed (research gap G3).")
        method = st.selectbox("Demand-level method", list(DEMAND_METHODS), format_func=DEMAND_METHODS.get,
                              key="sidebar_demand_method")
    with st.sidebar.expander("Filters"):
        min_d, max_d = clean_df["date"].min().date(), clean_df["date"].max().date()
        date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d,
                                   key="sidebar_date_range")
        all_days = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    if d in set(clean_df["day"])]
        days = st.multiselect("Weekdays", all_days, default=all_days, key="sidebar_weekdays")
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Pick both a start and an end date in the sidebar (Filters).")
        st.stop()
    return {"capacity": int(capacity), "alpha": float(alpha), "low": low_p, "high": high_p,
            "group_col": GRANULARITIES[granularity], "granularity": granularity, "method": method,
            "start": date_range[0], "end": date_range[1], "days": days}


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def build_navigation() -> tuple[dict, dict]:
    p = {
        "overview": st.Page(pages.overview, title="Overview", url_path="overview", default=True),
        "literature": st.Page(pages.literature, title="Literature Review", url_path="literature"),
        "gaps": st.Page(pages.gaps, title="Research Gaps", url_path="research-gaps"),
        "methodology": st.Page(pages.methodology, title="Methodology", url_path="methodology"),
        "data": st.Page(pages.data_eda, title="Data & EDA", url_path="data"),
        "rate": st.Page(pages.arrival_rate, title="Arrival Rate (λ̂)", url_path="arrival-rate"),
        "fit": st.Page(pages.goodness_of_fit, title="Goodness of Fit", url_path="goodness-of-fit"),
        "overload": st.Page(pages.overload, title="Overload Probability", url_path="overload"),
        "simulation": st.Page(pages.simulation, title="Simulation Check", url_path="simulation"),
        "results": st.Page(pages.results, title="Risk & Recommendations", url_path="results"),
        "quietest": st.Page(pages.quietest, title="Least-Crowded Time", url_path="least-crowded"),
        "limitations": st.Page(pages.limitations, title="Limitations & Future Scope", url_path="limitations"),
        "references": st.Page(pages.references, title="References & Downloads", url_path="references"),
        "syllabus": st.Page(pages.syllabus, title="Syllabus Map", url_path="syllabus"),
        "forecast": st.Page(pages.forecast, title="Peak-Time Forecast", url_path="forecast"),
        "validation": st.Page(pages.validation, title="Forecast Validation", url_path="validation"),
    }
    sections = {
        "Project": [p["overview"], p["literature"], p["gaps"], p["syllabus"], p["methodology"]],
        "Analysis": [p["data"], p["rate"], p["fit"], p["overload"], p["simulation"]],
        "Forecast": [p["forecast"], p["validation"]],
        "Results": [p["results"], p["quietest"]],
        "Conclusion": [p["limitations"], p["references"]],
    }
    return p, sections


def main() -> None:
    ui.html_block(ui.CSS)
    page_objects, sections = build_navigation()
    current = st.navigation(sections, expanded=True)

    try:
        raw, source_label, synthetic = sidebar_dataset()
        clean_df, report = clean_cached(raw)
    except ValueError as err:  # includes EmptyDatasetError and missing columns
        st.error(f"Could not load the dataset: {err}")
        st.stop()
    minutes = slot_minutes(clean_df)
    params = sidebar_parameters(clean_df, minutes)

    df = filter_dataset(clean_df, params["start"], params["end"], params["days"])
    if df.empty:
        st.warning("No observations match the selected dates and weekdays. Widen the filters in the sidebar.")
        st.stop()
    try:
        result = pipeline_cached(df, params["capacity"], params["alpha"], params["low"], params["high"],
                                 params["group_col"], params["method"])
    except ValueError as err:
        st.error(f"The analysis could not be completed: {err}")
        st.stop()

    periods = result.table["period"].tolist()
    st.session_state["_periods"] = periods
    if st.session_state.get("period") not in periods:
        st.session_state["period"] = result.kpis["max_overload_period"]
    st.sidebar.markdown("**Display**")
    period = st.sidebar.selectbox("Slot to inspect", periods, key="period", format_func=ui.slot,
                                  help="Used by the detailed views; you can also click a bar in the charts.")
    st.sidebar.toggle("Show reading notes", value=True, key="explain",
                      help="Notes beside each chart explaining how to read it.")

    ui.CTX.__dict__.update(raw_label=source_label, synthetic=synthetic, report=report, clean_df=clean_df, df=df,
                           params=params, result=result, period=period, minutes=minutes, pages=page_objects)
    current.run()


main()
