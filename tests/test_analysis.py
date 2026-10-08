"""
Unit tests for analysis.py and utils.py.
Covers:
  - Data cleaning edge cases (missing counts, negative/float counts, invalid dates, duplicates)
  - Empty dataset error handling
  - Group summarization (mean, variance, lambda, dispersion)
  - Demand level classification (tertile vs z-score)
  - End-to-end pipeline execution and KPI calculation
  - Recommendation engine
"""

import math
import numpy as np
import pandas as pd
import pytest

from utils import (
    clean_dataset,
    EmptyDatasetError,
    DEFAULT_CAPACITY,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_HIGH_THRESHOLD,
)
from analysis import (
    summarize_by_group,
    classify_demand_levels,
    add_overload_columns,
    run_pipeline,
    generate_recommendations,
    contiguous_windows,
    window_label,
)
from data_generator import generate_dataset


class TestDataCleaningAndValidation:
    def test_clean_valid_data(self):
        raw = generate_dataset(n_days=5, seed=42, missing_rate=0.0)
        clean_df, report = clean_dataset(raw)
        assert len(clean_df) == 5 * 8
        assert report["rows_in"] == report["rows_out"]
        assert report["missing_arrivals"] == 0
        assert report["invalid_arrivals"] == 0
        assert report["duplicates"] == 0

    def test_clean_removes_invalid_and_missing_rows(self):
        raw = pd.DataFrame([
            {"date": "2026-01-05", "time_interval": "08:00-08:15", "arrivals": 12},
            {"date": "2026-01-05", "time_interval": "08:15-08:30", "arrivals": None},      # missing
            {"date": "2026-01-05", "time_interval": "08:30-08:45", "arrivals": -3},        # negative
            {"date": "2026-01-05", "time_interval": "08:45-09:00", "arrivals": 14.5},      # fractional
            {"date": "not_a_date", "time_interval": "09:00-09:15", "arrivals": 10},        # bad date
            {"date": "2026-01-05", "time_interval": "invalid_time", "arrivals": 10},       # bad interval
            {"date": "2026-01-05", "time_interval": "08:00-08:15", "arrivals": 15},        # duplicate
        ])
        clean_df, report = clean_dataset(raw)
        assert len(clean_df) == 1
        assert report["rows_in"] == 7
        assert report["rows_out"] == 1
        assert report["missing_arrivals"] == 1
        assert report["invalid_arrivals"] == 2
        assert report["invalid_dates"] == 1
        assert report["invalid_intervals"] == 1
        assert report["duplicates"] == 1

    def test_empty_dataset_handling(self):
        with pytest.raises(EmptyDatasetError):
            clean_dataset(pd.DataFrame())

        empty_raw = pd.DataFrame(columns=["date", "time_interval", "arrivals"])
        with pytest.raises(EmptyDatasetError):
            clean_dataset(empty_raw)

    def test_dataset_with_only_invalid_rows(self):
        raw = pd.DataFrame([
            {"date": "2026-01-05", "time_interval": "08:00-08:15", "arrivals": -5},
            {"date": "2026-01-05", "time_interval": "08:15-08:30", "arrivals": np.nan},
        ])
        with pytest.raises(EmptyDatasetError):
            clean_dataset(raw)


class TestAnalysisPipeline:
    @pytest.fixture
    def sample_clean_df(self):
        # 30 days is the recommended minimum in experimental design
        raw = generate_dataset(n_days=30, seed=42, missing_rate=0.0)
        clean_df, _ = clean_dataset(raw)
        return clean_df

    def test_summarize_by_group(self, sample_clean_df):
        summary = summarize_by_group(sample_clean_df, group_col="time_interval")
        assert len(summary) == 8
        assert "lambda_hat" in summary.columns
        assert "variance" in summary.columns
        assert "var_mean_ratio" in summary.columns
        assert (summary["n"] == 30).all()

    def test_classify_demand_levels_tertile(self):
        lambdas = pd.Series([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        levels, info = classify_demand_levels(lambdas, method="tertile")
        assert info["method"] == "tertile"
        assert (levels == "Low").sum() >= 2
        assert (levels == "High").sum() >= 2

    def test_classify_demand_levels_zscore(self):
        lambdas = pd.Series([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        levels, info = classify_demand_levels(lambdas, method="zscore")
        assert info["method"] == "zscore"
        assert set(levels).issubset({"Low", "Medium", "High"})

    def test_add_overload_columns(self, sample_clean_df):
        summary = summarize_by_group(sample_clean_df, group_col="time_interval")
        with_overload = add_overload_columns(summary, capacity=30, low=0.20, high=0.50)
        assert "overload_prob" in with_overload.columns
        assert "risk" in with_overload.columns
        assert set(with_overload["risk"]).issubset({"LOW", "MEDIUM", "HIGH"})
        assert (with_overload["overload_prob"] >= 0.0).all()
        assert (with_overload["overload_prob"] <= 1.0).all()

    def test_run_pipeline_end_to_end(self, sample_clean_df):
        res = run_pipeline(
            sample_clean_df,
            capacity=30,
            alpha=0.05,
            low=0.20,
            high=0.50,
            group_col="time_interval",
            demand_method="tertile",
        )
        assert len(res.table) == 8
        assert len(res.gof_table) == 8
        assert res.kpis["total_observations"] == len(sample_clean_df)
        assert res.kpis["peak_period"] in set(sample_clean_df["time_interval"])
        assert len(res.recommendations) > 0
        assert res.gof_summary["tested"] == 8

    def test_hourly_pipeline(self, sample_clean_df):
        res = run_pipeline(
            sample_clean_df,
            capacity=30,
            alpha=0.05,
            low=0.20,
            high=0.50,
            group_col="hour_block",
        )
        assert len(res.table) == 3
        assert len(res.gof_table) == 3

    def test_small_sample_gof_graceful_handling(self):
        # 5 days: n=5 is too small for chi-square with min_expected=5, must not crash
        raw = generate_dataset(n_days=5, seed=42, missing_rate=0.0)
        clean_df, _ = clean_dataset(raw)
        res = run_pipeline(clean_df, capacity=30)
        assert len(res.gof_table) == 8
        # Valid should be False or decision should be 'Test not valid'
        assert (res.gof_table["valid"] == False).all()
        assert res.gof_summary["tested"] == 0

    def test_contiguous_windows_helper(self):
        periods = ["08:00-08:15", "08:15-08:30", "08:30-08:45", "09:00-09:15", "09:15-09:30"]
        flags = [True, True, False, True, True]
        windows = contiguous_windows(periods, flags)
        assert len(windows) == 2
        assert windows[0] == ["08:00-08:15", "08:15-08:30"]
        assert windows[1] == ["09:00-09:15", "09:15-09:30"]
        assert window_label(windows[0]) == "08:00–08:30"
