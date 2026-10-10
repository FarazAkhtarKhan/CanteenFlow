"""
Unit tests for the quietest-time analysis and the Monte Carlo check.
Covers:
  - Exact conditional test comparing two Poisson rates
  - Quietest-slot ranking and significance on the bundled dataset
  - Window search and comfort probabilities P(X <= k)
  - Weekday x slot rate table
  - Monte Carlo overload frequency converging to P(X > C)
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis import (comfort_probabilities, compare_poisson_rates, quietest_in_window, quietest_slots,
                      run_pipeline, weekday_rates)
from data_generator import generate_dataset
from poisson_model import overload_probability, simulate_overload_frequency
from utils import clean_dataset


@pytest.fixture(scope="module")
def bundled():
    df, _ = clean_dataset(generate_dataset(n_days=60, seed=42))
    return df, run_pipeline(df, capacity=30)


class TestComparePoissonRates:
    def test_equal_rates_are_not_significant(self):
        assert compare_poisson_rates(500, 50, 500, 50) > 0.4

    def test_clearly_quieter_rate_is_significant(self):
        assert compare_poisson_rates(300, 30, 600, 30) < 1e-6

    def test_matches_binomial_formula(self):
        expected = stats.binom.cdf(40, 40 + 70, 10 / (10 + 12))
        assert compare_poisson_rates(40, 10, 70, 12) == pytest.approx(expected)

    def test_zero_totals_give_p_of_one(self):
        assert compare_poisson_rates(0, 5, 0, 5) == 1.0

    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError):
            compare_poisson_rates(10, 0, 10, 5)


class TestQuietestSlots:
    def test_best_is_lowest_lambda(self, bundled):
        _, result = bundled
        q = result.quiet
        assert q["best"] == result.table.loc[result.table["lambda_hat"].idxmin(), "period"]
        assert q["ranking"]["lambda_hat"].is_monotonic_increasing
        assert q["best_ci"][0] <= q["best_lambda"] <= q["best_ci"][1]

    def test_end_of_service_slot_is_significantly_quietest(self, bundled):
        _, result = bundled
        assert result.quiet["best"] == "14:15-14:30"
        assert result.quiet["significant"]

    def test_single_period_has_no_runner_up(self, bundled):
        _, result = bundled
        q = quietest_slots(result.table.iloc[[0]])
        assert q["runner_up"] is None and not q["significant"]

    def test_recommendations_mention_quietest(self, bundled):
        _, result = bundled
        assert any("Quietest time to visit" in rec for rec in result.recommendations)


class TestPlanning:
    def test_window_restricts_choice(self, bundled):
        _, result = bundled
        pick = quietest_in_window(result.table, "12:30", "13:30")
        assert pick["period"] == "12:30-12:45"

    def test_window_without_full_slot_returns_none(self, bundled):
        _, result = bundled
        assert quietest_in_window(result.table, "13:05", "13:10") is None

    def test_comfort_probabilities_are_poisson_cdf(self, bundled):
        _, result = bundled
        probs = comfort_probabilities(result.table, 20)
        lam = result.table.set_index("period").loc["13:15-13:30", "lambda_hat"]
        assert probs["13:15-13:30"] == pytest.approx(stats.poisson.cdf(20, lam))
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_negative_comfort_level_rejected(self, bundled):
        _, result = bundled
        with pytest.raises(ValueError):
            comfort_probabilities(result.table, -1)

    def test_weekday_rates_shape(self, bundled):
        df, _ = bundled
        rates = weekday_rates(df)
        assert list(rates.index) == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        assert rates.shape[1] == 8
        assert not rates.isna().any().any()


class TestMonteCarlo:
    def test_running_frequency_converges(self):
        lam, capacity = 30.0, 30
        _, running = simulate_overload_frequency(lam, capacity, 20_000, seed=1)
        assert running[-1] == pytest.approx(overload_probability(lam, capacity), abs=0.015)

    def test_reproducible_with_seed(self):
        a = simulate_overload_frequency(20.0, 25, 500, seed=3)[0]
        b = simulate_overload_frequency(20.0, 25, 500, seed=3)[0]
        assert np.array_equal(a, b)

    def test_rejects_zero_days(self):
        with pytest.raises(ValueError):
            simulate_overload_frequency(20.0, 25, 0)
