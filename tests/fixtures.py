"""
tests/fixtures.py
-----------------
Random Poisson test data with known parameters, used only by the unit tests.

The tests need data whose true rates, weekday effects and day-type effects are known, so that they can check
that the estimators recover them (for example, that the regression finds a known exam-day multiplier). The
dashboard, the paper and the report never use this module; they read the recorded arrivals in
data/canteen_arrivals.csv.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from utils import DAY_NAMES, DEFAULT_CAPACITY, period_label_for_hour, validate_capacity

# --- Canteen operating schedule (12:30 PM to 2:30 PM) -----------------------
DEFAULT_START_TIME = "12:30"
DEFAULT_END_TIME = "14:30"
INTERVAL_MINUTES = 15                  # bundled dataset; 5 and 10 minutes are also supported
SUPPORTED_INTERVALS = (5, 10, 15)
DEFAULT_SEED = 42
DEFAULT_N_DAYS = 90                    # one 18-week semester of operating data
DEFAULT_START_DATE = date(2026, 1, 5)  # a Monday
DEFAULT_MISSING_RATE = 0.005

# --- Baseline lunch demand profile -----------------------------------------
# Centered around the 1:15 PM (13:15) peak lunch rush
BACKGROUND_RATE = 14.0  # arrivals per 15 min at start (12:30) and end (14:15)
DEMAND_BUMPS = [
    # (period_name, center [decimal hour], spread [hours], peak_extra_arrivals)
    ("lunch rush", 13.25, 0.55, 21.0),
]

# --- Day-to-day variation ----------------------------------------------------
DAY_FACTOR_CV = 0.05
WEEKDAY_EFFECT = {"Monday": 1.02, "Tuesday": 1.00, "Wednesday": 1.01, "Thursday": 1.00, "Friday": 0.94}

# --- Special days ------------------------------------------------------------
EXAM_EFFECT = 0.85        # fewer classes, fewer students on campus at lunch
SEMESTER_WEEKS = 18       # working weeks per semester
EXAM_WEEKS = (8, 17)      # mid-semester (week 9) and end-semester (week 18) exams, counted from 0


def is_exam_day(day_index: int) -> bool:
    """True for working days that fall in an exam week (day_index counts working days from 0)."""
    return (day_index // 5) % SEMESTER_WEEKS in EXAM_WEEKS


def build_time_intervals(start_time: str = DEFAULT_START_TIME,
                         end_time: str = DEFAULT_END_TIME,
                         minutes: int = INTERVAL_MINUTES,
                         open_hour: int | None = None,
                         close_hour: int | None = None) -> list[tuple[str, int, float]]:
    """Return [(label 'HH:MM-HH:MM', hour, midpoint in decimal hours), ...].

    Defaults to 12:30 PM – 2:30 PM (8 intervals of 15 minutes).
    Supports legacy open_hour/close_hour parameters for backward compatibility.
    """
    if open_hour is not None and close_hour is not None:
        start_min = open_hour * 60
        end_min = close_hour * 60
    else:
        h1, m1 = (int(p) for p in start_time.split(":"))
        h2, m2 = (int(p) for p in end_time.split(":"))
        start_min = h1 * 60 + m1
        end_min = h2 * 60 + m2

    if not 0 <= start_min < end_min <= 24 * 60:
        raise ValueError("Start time must be before end time.")
    if minutes <= 0:
        raise ValueError("Interval length must be positive.")

    intervals = []
    for start in range(start_min, end_min, minutes):
        end = start + minutes
        label = f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"
        intervals.append((label, start // 60, (start + minutes / 2) / 60))
    return intervals


def baseline_lambda(midpoints_hours) -> np.ndarray:
    """Expected arrivals per interval from the assumed demand profile.

    lambda(t) = background + sum_k A_k * exp(-(t - mu_k)^2 / (2 * sigma_k^2))
    """
    t = np.asarray(midpoints_hours, dtype=float)
    rate = np.full_like(t, BACKGROUND_RATE)
    for _name, centre, spread, amplitude in DEMAND_BUMPS:
        rate += amplitude * np.exp(-((t - centre) ** 2) / (2 * spread ** 2))
    return rate


def working_days(start: date, n_days: int) -> list[date]:
    """The first `n_days` Monday-Friday dates on/after `start`."""
    days, current = [], start
    while len(days) < n_days:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def default_capacity_for(interval_minutes: int) -> int:
    """Capacity scaled to the slot length (30 per 15 minutes -> 20 per 10 minutes, 10 per 5 minutes)."""
    return max(1, round(DEFAULT_CAPACITY * interval_minutes / INTERVAL_MINUTES))


def generate_dataset(n_days: int = DEFAULT_N_DAYS, seed: int = DEFAULT_SEED,
                     start_date: date = DEFAULT_START_DATE, capacity: int = DEFAULT_CAPACITY,
                     missing_rate: float = DEFAULT_MISSING_RATE,
                     interval_minutes: int = INTERVAL_MINUTES) -> pd.DataFrame:
    """Generate the synthetic arrivals dataset as a DataFrame.

    `interval_minutes` sets the slot length (5, 10 or 15). The demand profile is defined
    per 15 minutes, so expected counts are scaled by interval_minutes / 15.

    Columns: date, day, day_type, time_interval, hour, arrivals, period, capacity
    """
    if interval_minutes not in SUPPORTED_INTERVALS:
        raise ValueError(f"Slot length must be one of {SUPPORTED_INTERVALS} minutes (got {interval_minutes!r}).")
    if isinstance(n_days, bool) or not isinstance(n_days, (int, np.integer)) or n_days < 1:
        raise ValueError(f"Number of days must be a positive integer (got {n_days!r}).")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")
    capacity = validate_capacity(capacity)

    rng = np.random.default_rng(seed)
    intervals = build_time_intervals(minutes=interval_minutes)
    base_rates = baseline_lambda([mid for _, _, mid in intervals]) * interval_minutes / INTERVAL_MINUTES
    gamma_shape = 1.0 / DAY_FACTOR_CV ** 2  # Gamma(k, 1/k) has mean 1 and CV = 1/sqrt(k)

    rows = []
    for index, day in enumerate(working_days(start_date, int(n_days))):
        day_name = DAY_NAMES[day.weekday()]
        exam = is_exam_day(index)
        day_factor = rng.gamma(gamma_shape, 1.0 / gamma_shape) * WEEKDAY_EFFECT[day_name]
        if exam:
            day_factor *= EXAM_EFFECT
        counts = rng.poisson(base_rates * day_factor)
        for (label, hour, _mid), count in zip(intervals, counts):
            rows.append({
                "date": day.isoformat(),
                "day": day_name,
                "day_type": "Exam" if exam else "Regular",
                "time_interval": label,
                "hour": hour,
                "arrivals": int(count),
                "period": period_label_for_hour(hour),
                "capacity": capacity,
            })

    df = pd.DataFrame(rows)
    df["arrivals"] = df["arrivals"].astype("Int64")  # nullable int so gaps stay integers
    if missing_rate > 0:
        gaps = rng.random(len(df)) < missing_rate
        df.loc[gaps, "arrivals"] = pd.NA
    return df
