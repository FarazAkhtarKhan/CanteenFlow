"""
utils.py
--------
Shared constants, input validation and the data-cleaning step of the pipeline.

Pipeline position:
    Data Collection -> **Data Cleaning** -> Exploratory Analysis -> ...
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-wide defaults (all of these can be changed from the dashboard)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "canteen_arrivals.csv"

# Arrivals the canteen can reasonably handle in ONE 15-minute interval.
# Assumption: 2 service counters x ~15 students each per 15 minutes.
DEFAULT_CAPACITY = 30
DEFAULT_ALPHA = 0.05
# Project-defined decision thresholds (NOT universal scientific standards).
DEFAULT_LOW_THRESHOLD = 0.20   # P(X > C) <  20%        -> LOW
DEFAULT_HIGH_THRESHOLD = 0.50  # P(X > C) >  50%        -> HIGH, otherwise MEDIUM

REQUIRED_COLUMNS = ["date", "time_interval", "arrivals"]
SCHEMA_COLUMNS = ["date", "day", "time_interval", "hour", "arrivals", "period", "capacity"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_INTERVAL_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


class EmptyDatasetError(ValueError):
    """Raised when a dataset has no usable rows."""


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def period_label_for_hour(hour: int) -> str:
    """Descriptive clock-time label for an hour of the day.

    These labels only describe the time of day. They are NOT demand levels;
    demand levels are computed from the data in analysis.py.
    """
    if hour < 10:
        return "Breakfast"
    if hour < 12:
        return "Mid-Morning"
    if hour < 15:
        return "Lunch"
    if hour < 17:
        return "Afternoon"
    return "Evening"


def hour_block_label(hour: int) -> str:
    """Return a one-hour block label, e.g. 12 -> '12:00-13:00'."""
    return f"{int(hour):02d}:00-{int(hour) + 1:02d}:00"


def normalize_interval_label(label: object) -> str | None:
    """Normalise an interval label to 'HH:MM-HH:MM' or return None if invalid."""
    if not isinstance(label, str):
        return None
    match = _INTERVAL_PATTERN.match(label)
    if match is None:
        return None
    h1, m1, h2, m2 = (int(part) for part in match.groups())
    if not (0 <= h1 <= 23 and 0 <= h2 <= 24 and m1 < 60 and m2 < 60):
        return None
    if h2 * 60 + m2 <= h1 * 60 + m1:  # interval must move forward in time
        return None
    return f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_capacity(capacity: object) -> int:
    """Validate the service capacity C and return it as an int.

    C is a count of arrivals per interval, so it must be a whole number >= 1.
    """
    if capacity is None or isinstance(capacity, bool):
        raise ValueError("Capacity must be a positive whole number (got None/bool).")
    try:
        value = float(capacity)
    except (TypeError, ValueError):
        raise ValueError(f"Capacity must be a positive whole number (got {capacity!r}).") from None
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Capacity must be a finite number.")
    if not value.is_integer():
        raise ValueError(f"Capacity must be a whole number of arrivals (got {capacity}).")
    if value < 1:
        raise ValueError(f"Capacity must be at least 1 arrival per interval (got {capacity}).")
    return int(value)


def validate_thresholds(low: float, high: float) -> tuple[float, float]:
    """Validate the LOW/HIGH risk thresholds (as probabilities in [0, 1])."""
    for name, value in (("low", low), ("high", high)):
        if value is None or not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"The {name} threshold must be a number.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"The {name} threshold must be between 0 and 1 (got {value}).")
    if low >= high:
        raise ValueError(f"The low threshold ({low}) must be smaller than the high threshold ({high}).")
    return float(low), float(high)


def validate_alpha(alpha: float) -> float:
    """Validate a significance level, 0 < alpha < 1."""
    if alpha is None or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise ValueError(f"Significance level alpha must be between 0 and 1 (got {alpha}).")
    return float(alpha)


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def load_dataset(source=DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read a CSV file (path or file-like object) into a DataFrame."""
    try:
        return pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise EmptyDatasetError("The CSV file is empty.") from None


def clean_dataset(raw: pd.DataFrame, default_capacity: int = DEFAULT_CAPACITY) -> tuple[pd.DataFrame, dict]:
    """Validate and clean an arrivals dataset.

    Rules (each is counted in the returned report):
      * rows with a missing / non-numeric arrival count are dropped
        (imputing a count, e.g. with the mean, would artificially shrink the
        variance and bias the mean-vs-variance check);
      * negative or non-integer arrival counts are dropped (counts must be 0,1,2,...);
      * rows with an unreadable date or time interval are dropped;
      * duplicate (date, time_interval) records are dropped, keeping the first;
      * `day`, `hour` and `hour_block` are always re-derived from date/time so they are consistent;
      * `period` and `capacity` are filled in if missing.

    Returns (clean DataFrame, report dict).
    """
    if raw is None or len(raw) == 0:
        raise EmptyDatasetError("The dataset is empty - there are no rows to analyse.")

    df = raw.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing_cols)}.")

    report = {"rows_in": int(len(df))}

    # Arrivals: missing / non-numeric
    arrivals = pd.to_numeric(df["arrivals"], errors="coerce")
    missing_mask = arrivals.isna()
    report["missing_arrivals"] = int(missing_mask.sum())

    # Arrivals: negative or fractional values are impossible for a count
    invalid_mask = ~missing_mask & ((arrivals < 0) | (arrivals % 1 != 0))
    report["invalid_arrivals"] = int(invalid_mask.sum())

    # Dates and interval labels
    dates = pd.to_datetime(df["date"], errors="coerce")
    bad_date_mask = dates.isna() & ~missing_mask & ~invalid_mask
    report["invalid_dates"] = int(bad_date_mask.sum())

    intervals = df["time_interval"].map(normalize_interval_label)
    bad_interval_mask = intervals.isna() & ~missing_mask & ~invalid_mask & ~bad_date_mask
    report["invalid_intervals"] = int(bad_interval_mask.sum())

    keep = ~(missing_mask | invalid_mask | bad_date_mask | bad_interval_mask)
    df = df.loc[keep].copy()
    df["arrivals"] = arrivals[keep].astype(int)
    df["date"] = dates[keep].dt.normalize()
    df["time_interval"] = intervals[keep]

    # Duplicate records of the same interval on the same day
    dup_mask = df.duplicated(subset=["date", "time_interval"], keep="first")
    report["duplicates"] = int(dup_mask.sum())
    df = df.loc[~dup_mask]

    if df.empty:
        raise EmptyDatasetError("No valid rows remain after cleaning the dataset.")

    # Derived / filled columns
    df["day"] = df["date"].dt.dayofweek.map(lambda i: DAY_NAMES[i])
    df["hour"] = df["time_interval"].str.slice(0, 2).astype(int)
    df["hour_block"] = df["hour"].map(hour_block_label)
    if "period" not in df.columns:
        df["period"] = np.nan
    df["period"] = df["period"].where(df["period"].notna(), df["hour"].map(period_label_for_hour))
    if "capacity" not in df.columns:
        df["capacity"] = default_capacity
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(default_capacity).astype(int)

    df = df.sort_values(["date", "time_interval"]).reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["rows_removed"] = report["rows_in"] - report["rows_out"]
    columns = SCHEMA_COLUMNS + ["hour_block"]
    return df[columns], report


def slot_minutes(df: pd.DataFrame, fallback: int = 15) -> int:
    """Most common slot length in minutes, read from the 'HH:MM-HH:MM' labels."""
    if df is None or df.empty or "time_interval" not in df.columns:
        return fallback
    labels = pd.Series(df["time_interval"].unique())
    starts = labels.str.slice(0, 2).astype(int) * 60 + labels.str.slice(3, 5).astype(int)
    ends = labels.str.slice(6, 8).astype(int) * 60 + labels.str.slice(9, 11).astype(int)
    lengths = (ends - starts)[(ends - starts) > 0]
    return int(lengths.mode().iloc[0]) if not lengths.empty else fallback


def dataset_default_capacity(df: pd.DataFrame, fallback: int = DEFAULT_CAPACITY) -> int:
    """Most common capacity value in the dataset (used only as the UI default)."""
    if df is None or df.empty or "capacity" not in df.columns:
        return fallback
    mode = df["capacity"].mode()
    try:
        return validate_capacity(mode.iloc[0]) if not mode.empty else fallback
    except ValueError:
        return fallback
