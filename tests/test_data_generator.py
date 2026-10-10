"""
Unit tests for data_generator.py.
Covers:
  - Dataset generation schema and data types
  - Seed reproducibility (fixed seed gives identical results)
  - Configurable parameters (days, capacity, missing_rate)
  - Working days calculation and time interval format
"""

import numpy as np
import pandas as pd
import pytest

from data_generator import (
    generate_dataset,
    build_time_intervals,
    working_days,
    baseline_lambda,
    DEFAULT_N_DAYS,
    DEFAULT_SEED,
)
from utils import SCHEMA_COLUMNS, validate_capacity


class TestDataGenerator:
    def test_build_time_intervals(self):
        # Default schedule: 12:30 to 14:30 -> 8 intervals of 15 minutes
        intervals = build_time_intervals()
        assert len(intervals) == 8
        first_label, first_hour, first_mid = intervals[0]
        assert first_label == "12:30-12:45"
        assert first_hour == 12
        assert math_check(first_mid, 12.625)

        last_label, last_hour, last_mid = intervals[-1]
        assert last_label == "14:15-14:30"
        assert last_hour == 14

        # Legacy backward compatibility check
        legacy = build_time_intervals(open_hour=8, close_hour=18, minutes=15)
        assert len(legacy) == 40

    def test_schema_and_columns(self):
        df = generate_dataset(n_days=2, seed=42, missing_rate=0.0)
        # 2 days * 8 intervals = 16 rows
        assert len(df) == 16
        for col in SCHEMA_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"

    def test_seed_reproducibility(self):
        df1 = generate_dataset(n_days=5, seed=123, missing_rate=0.0)
        df2 = generate_dataset(n_days=5, seed=123, missing_rate=0.0)
        pd.testing.assert_frame_equal(df1, df2)

        # Different seed should produce different arrivals
        df3 = generate_dataset(n_days=5, seed=999, missing_rate=0.0)
        assert not df1["arrivals"].equals(df3["arrivals"])

    def test_configurable_days_and_capacity(self):
        df = generate_dataset(n_days=10, seed=42, capacity=45, missing_rate=0.0)
        assert len(df) == 10 * 8
        assert (df["capacity"] == 45).all()
        assert df["date"].nunique() == 10

    def test_missing_rate_creates_nulls(self):
        # With missing_rate = 0.1, approx 10% of arrivals should be NA
        df = generate_dataset(n_days=20, seed=42, missing_rate=0.10)
        missing_count = df["arrivals"].isna().sum()
        total_rows = len(df)
        ratio = missing_count / total_rows
        assert 0.05 < ratio < 0.15

    def test_invalid_days_or_capacity(self):
        with pytest.raises(ValueError):
            generate_dataset(n_days=0)
        with pytest.raises(ValueError):
            generate_dataset(n_days=-5)
        with pytest.raises(ValueError):
            generate_dataset(capacity=0)


def math_check(val, expected):
    return abs(val - expected) < 1e-6


class TestSlotLength:
    @pytest.mark.parametrize("minutes, slots", [(5, 24), (10, 12), (15, 8)])
    def test_slot_count_and_length(self, minutes, slots):
        from utils import clean_dataset, slot_minutes
        df, _ = clean_dataset(generate_dataset(n_days=10, seed=1, interval_minutes=minutes, missing_rate=0.0))
        assert df["time_interval"].nunique() == slots
        assert slot_minutes(df) == minutes

    def test_rates_scale_with_slot_length(self):
        from data_generator import default_capacity_for
        ten = generate_dataset(n_days=200, seed=3, interval_minutes=10, missing_rate=0.0)
        fifteen = generate_dataset(n_days=200, seed=3, interval_minutes=15, missing_rate=0.0)
        per_minute_10 = ten["arrivals"].astype(float).sum() / (len(ten) * 10)
        per_minute_15 = fifteen["arrivals"].astype(float).sum() / (len(fifteen) * 15)
        assert per_minute_10 == pytest.approx(per_minute_15, rel=0.03)
        assert default_capacity_for(10) == 20 and default_capacity_for(5) == 10

    def test_unsupported_slot_length_rejected(self):
        with pytest.raises(ValueError):
            generate_dataset(n_days=5, interval_minutes=7)
