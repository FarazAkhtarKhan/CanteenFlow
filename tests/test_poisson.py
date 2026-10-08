"""
Unit tests for poisson_model.py and goodness_of_fit.py.
Covers:
  - Poisson PMF calculation (manual formula vs SciPy)
  - Lambda estimation
  - Mean calculation
  - Variance calculation
  - Overload probability calculation (manual vs SciPy)
  - Risk classification (threshold edge cases and validation)
  - Handling invalid capacity and inputs
  - Chi-Square goodness-of-fit and category merging
"""

import math
import numpy as np
import pytest
from scipy import stats

from poisson_model import (
    estimate_lambda,
    sample_mean,
    sample_variance,
    poisson_pmf_manual,
    poisson_pmf,
    poisson_cdf_manual,
    overload_probability_manual,
    overload_probability,
    classify_risk,
    lambda_confidence_interval,
    required_capacity,
    to_count_array,
)
from goodness_of_fit import (
    chi_square_gof,
    dispersion_test,
    build_frequency_table,
    MIN_EXPECTED_FREQUENCY,
)
from utils import validate_capacity, validate_thresholds, validate_alpha


class TestPoissonCore:
    def test_lambda_estimation(self):
        counts = [10, 12, 14, 16, 18]
        expected_mean = 14.0
        assert math.isclose(estimate_lambda(counts), expected_mean)
        assert math.isclose(sample_mean(counts), expected_mean)

    def test_sample_variance(self):
        counts = [10, 12, 14, 16, 18]
        # s^2 with ddof=1: sum((x - 14)^2)/4 = (16 + 4 + 0 + 4 + 16)/4 = 40/4 = 10.0
        assert math.isclose(sample_variance(counts), 10.0)

    def test_variance_single_element(self):
        assert math.isnan(sample_variance([5]))

    def test_poisson_pmf_manual_matches_scipy(self):
        test_cases = [
            (0, 5.0),
            (3, 5.0),
            (5, 5.0),
            (10, 5.0),
            (25, 25.0),
            (30, 24.5),
        ]
        for x, lam in test_cases:
            manual = poisson_pmf_manual(x, lam)
            scipy_val = stats.poisson.pmf(x, lam)
            assert math.isclose(manual, scipy_val, rel_tol=1e-7, abs_tol=1e-9)
            assert math.isclose(poisson_pmf(x, lam), scipy_val, rel_tol=1e-7)

    def test_poisson_pmf_boundary_zero(self):
        assert math.isclose(poisson_pmf_manual(0, 0.0), 1.0)
        assert math.isclose(poisson_pmf_manual(1, 0.0), 0.0)

    def test_poisson_cdf_manual(self):
        lam = 12.0
        c = 15
        manual_cdf = poisson_cdf_manual(c, lam)
        scipy_cdf = stats.poisson.cdf(c, lam)
        assert math.isclose(manual_cdf, scipy_cdf, rel_tol=1e-6)

    def test_overload_probability_manual_and_scipy(self):
        lam = 25.0
        capacity = 30
        p_manual = overload_probability_manual(lam, capacity)
        p_scipy = overload_probability(lam, capacity)
        expected_sf = stats.poisson.sf(capacity, lam)

        assert math.isclose(p_manual, expected_sf, rel_tol=1e-6)
        assert math.isclose(p_scipy, expected_sf, rel_tol=1e-7)
        # For lambda=25, capacity=30, P(X > 30) is approx 13.67%
        assert 0.10 < p_manual < 0.20

    def test_confidence_interval(self):
        total = 600
        n = 40
        low, high = lambda_confidence_interval(total, n, confidence=0.95)
        lam_hat = total / n
        assert low < lam_hat < high
        assert low > 0

    def test_required_capacity(self):
        lam = 30.0
        # If max allowed overload is 10%, capacity must ensure P(X > C) < 0.10
        cap = required_capacity(lam, 0.10)
        assert stats.poisson.sf(cap, lam) < 0.10
        assert stats.poisson.sf(cap - 1, lam) >= 0.10


class TestRiskClassification:
    def test_risk_levels(self):
        # Default thresholds: low=0.20, high=0.50
        assert classify_risk(0.05, 0.20, 0.50) == "LOW"
        assert classify_risk(0.199, 0.20, 0.50) == "LOW"
        assert classify_risk(0.20, 0.20, 0.50) == "MEDIUM"
        assert classify_risk(0.35, 0.20, 0.50) == "MEDIUM"
        assert classify_risk(0.50, 0.20, 0.50) == "MEDIUM"
        assert classify_risk(0.501, 0.20, 0.50) == "HIGH"
        assert classify_risk(0.95, 0.20, 0.50) == "HIGH"

    def test_configurable_thresholds(self):
        # Custom thresholds: low=0.10, high=0.30
        assert classify_risk(0.08, 0.10, 0.30) == "LOW"
        assert classify_risk(0.15, 0.10, 0.30) == "MEDIUM"
        assert classify_risk(0.35, 0.10, 0.30) == "HIGH"


class TestValidationAndErrorHandling:
    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="at least 1 arrival"):
            validate_capacity(0)
        with pytest.raises(ValueError, match="at least 1 arrival"):
            validate_capacity(-5)
        with pytest.raises(ValueError, match="whole number"):
            validate_capacity(25.5)
        with pytest.raises(ValueError):
            validate_capacity("not_a_number")
        with pytest.raises(ValueError):
            validate_capacity(None)

    def test_valid_capacity(self):
        assert validate_capacity(30) == 30
        assert validate_capacity("25") == 25
        assert validate_capacity(40.0) == 40

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="smaller than the high"):
            validate_thresholds(0.5, 0.2)
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_thresholds(-0.1, 0.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_thresholds(0.2, 1.5)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            validate_alpha(0.0)
        with pytest.raises(ValueError):
            validate_alpha(1.0)
        with pytest.raises(ValueError):
            validate_alpha(-0.05)

    def test_empty_arrivals_array(self):
        with pytest.raises(ValueError, match="empty set"):
            to_count_array([])

    def test_negative_arrivals(self):
        with pytest.raises(ValueError, match="negative"):
            to_count_array([5, 10, -2])


class TestGoodnessOfFit:
    def test_chi_square_poisson_fit_null_hypothesis(self):
        # Generate a large sample from a true Poisson(15) distribution
        rng = np.random.default_rng(12345)
        sample = rng.poisson(lam=15.0, size=500)

        result = chi_square_gof(sample, alpha=0.05)
        assert result.valid is True
        assert result.df >= 1
        assert not math.isnan(result.chi_square)
        assert 0.0 <= result.p_value <= 1.0
        # For a true Poisson draw of size 500, we should typically fail to reject H0
        assert result.reject_h0 is False
        assert result.decision == "Fail to reject H0"

    def test_chi_square_detects_non_poisson(self):
        # Generate bimodal / non-Poisson mixture with heavy overdispersion
        rng = np.random.default_rng(54321)
        part1 = rng.poisson(lam=5.0, size=250)
        part2 = rng.poisson(lam=40.0, size=250)
        mixture = np.concatenate([part1, part2])

        result = chi_square_gof(mixture, alpha=0.05)
        assert result.valid is True
        assert result.reject_h0 is True
        assert result.decision == "Reject H0"
        assert result.p_value < 0.05

    def test_category_merging_respects_min_expected(self):
        rng = np.random.default_rng(999)
        sample = rng.poisson(lam=8.0, size=80)
        table, notes = build_frequency_table(sample, lam=float(sample.mean()), min_expected=MIN_EXPECTED_FREQUENCY)

        assert len(table) >= 3
        # Each category in the table should have expected frequency >= 5.0
        # (or be appropriately merged into the final group)
        for expected in table["expected"]:
            assert expected >= MIN_EXPECTED_FREQUENCY - 1e-6
        # Probabilities should sum to 1.0
        assert math.isclose(table["probability"].sum(), 1.0, rel_tol=1e-5)

    def test_dispersion_test(self):
        rng = np.random.default_rng(42)
        sample = rng.poisson(lam=10.0, size=100)
        disp = dispersion_test(sample, alpha=0.05)
        assert 0.7 < disp["index"] < 1.3
        assert disp["label"] in ("Consistent with Poisson", "Overdispersed", "Underdispersed")
