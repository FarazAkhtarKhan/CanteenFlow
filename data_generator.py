"""
data_generator.py
-----------------
Generates a **SYNTHETIC** dataset of student arrivals at a college canteen.

    !!! The data produced here is synthetically generated research/prototype data.
    !!! It was NOT collected from any real college canteen.

How the data is simulated
=========================
1. The operating day (08:00-18:00) is divided into fixed 15-minute intervals
   (40 intervals per day). The canteen is assumed to operate Monday-Friday.
2. A smooth *baseline demand profile* assigns an expected count lambda(t) to each
   interval: a constant background rate plus four bell-shaped "bumps" for typical
   campus routines (breakfast, mid-morning break, lunch, evening snacks).
   These are MODELLING ASSUMPTIONS used only to simulate data. The analysis
   code never sees this profile - it re-estimates lambda from the counts alone.
3. Day-to-day variation: every day gets a multiplicative attendance factor drawn
   from a Gamma distribution with mean 1 and a 5% coefficient of variation,
   plus a mild weekday effect (e.g. Fridays slightly quieter).
4. Each interval's arrival count is drawn with NumPy's Poisson generator:
       arrivals ~ Poisson(lambda(t) x day_factor)
5. A small fraction of records (default 0.5%) is blanked out to simulate
   recording gaps, so that the data-cleaning step has real work to do.

A fixed random seed makes the dataset exactly reproducible.

Usage (command line):
    python data_generator.py                 # 60 working days, seed 42
    python data_generator.py --days 90 --seed 7 --capacity 28
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DAY_NAMES, DEFAULT_CAPACITY, DEFAULT_DATA_PATH, period_label_for_hour, validate_capacity

# --- Canteen operating schedule (12:30 PM to 2:30 PM) -----------------------
DEFAULT_START_TIME = "12:30"
DEFAULT_END_TIME = "14:30"
INTERVAL_MINUTES = 15                  # bundled dataset; 5 and 10 minutes are also supported
SUPPORTED_INTERVALS = (5, 10, 15)
DEFAULT_SEED = 42
DEFAULT_N_DAYS = 60                    # 12 weeks of operating data
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

    Columns: date, day, time_interval, hour, arrivals, period, capacity
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
    for day in working_days(start_date, int(n_days)):
        day_name = DAY_NAMES[day.weekday()]
        day_factor = rng.gamma(gamma_shape, 1.0 / gamma_shape) * WEEKDAY_EFFECT[day_name]
        counts = rng.poisson(base_rates * day_factor)
        for (label, hour, _mid), count in zip(intervals, counts):
            rows.append({
                "date": day.isoformat(),
                "day": day_name,
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


def save_dataset(df: pd.DataFrame, path: Path | str = DEFAULT_DATA_PATH) -> Path:
    """Write the dataset to CSV (missing counts are written as empty cells)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic canteen arrivals dataset.")
    parser.add_argument("--days", type=int, default=DEFAULT_N_DAYS, help="number of working days")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--interval", type=int, default=INTERVAL_MINUTES, choices=SUPPORTED_INTERVALS,
                        help="slot length in minutes")
    parser.add_argument("--capacity", type=int, default=None, help="capacity per slot (default scales with --interval)")
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_PATH, help="output CSV path")
    args = parser.parse_args()

    capacity = args.capacity if args.capacity is not None else default_capacity_for(args.interval)
    df = generate_dataset(args.days, args.seed, capacity=capacity, missing_rate=args.missing_rate,
                          interval_minutes=args.interval)
    path = save_dataset(df, args.out)
    n_slots = len(build_time_intervals(minutes=args.interval))
    print(f"Saved {len(df)} records ({args.days} working days x {n_slots} {args.interval}-minute slots, "
          f"12:30 PM - 2:30 PM) to {path}")
    print(f"Missing count records: {int(df['arrivals'].isna().sum())}")


if __name__ == "__main__":
    main()
