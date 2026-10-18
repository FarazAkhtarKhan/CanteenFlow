"""
Unit tests for forecast.py (research gaps G7-G11).
Covers:
  - Poisson-Gamma (negative binomial) probabilities reduce to Poisson when Var(W) = 0
  - Regression recovers weekday and exam-day effects; prediction errors for unseen levels
  - Day-to-day variance estimate (method of moments) and the Pearson test
  - Peak-time probabilities, same-day Bayesian updating
  - Runs test, proportion z-test, Brier score
  - Rolling backtest uses no future data; summary tests are well formed
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import forecast as fc
from data_generator import EXAM_EFFECT, WEEKDAY_EFFECT, generate_dataset
from utils import clean_dataset


@pytest.fixture(scope="module")
def big_df():
    df, _ = clean_dataset(generate_dataset(n_days=360, seed=11, missing_rate=0.0))
    return df


@pytest.fixture(scope="module")
def semester_df():
    df, _ = clean_dataset(generate_dataset(seed=42))
    return df


def poisson_only_df(n_days=200, seed=3):
    """Pure Poisson counts (no day factor), same rates every day."""
    rng = np.random.default_rng(seed)
    rates = {"12:00-12:15": 10.0, "12:15-12:30": 25.0, "12:30-12:45": 18.0}
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    rows = [{"date": d, "time_interval": p, "arrivals": int(rng.poisson(r))} for d in dates for p, r in rates.items()]
    df, _ = clean_dataset(pd.DataFrame(rows))
    return df


class TestDistributions:
    def test_mixed_reduces_to_poisson(self):
        assert fc.overload_probability_mixed(25.0, 30, 0.0) == pytest.approx(stats.poisson.sf(30, 25.0))

    def test_negative_binomial_moments(self):
        lam, v = 20.0, 0.05
        n, p = fc._nb_params(lam, v)
        assert stats.nbinom.mean(n, p) == pytest.approx(lam)
        assert stats.nbinom.var(n, p) == pytest.approx(lam + v * lam ** 2)

    def test_extra_variance_fattens_the_tail(self):
        # Above the mean, day-to-day variation makes overload more likely
        assert fc.overload_probability_mixed(25.0, 35, 0.05) > fc.overload_probability_mixed(25.0, 35, 0.0)

    def test_prediction_interval_covers_mean(self):
        lo, hi = fc.predictive_interval(np.array([20.0]), 0.01)
        assert lo[0] < 20 < hi[0]


class TestRateModel:
    def test_recovers_effects(self, big_df):
        model = fc.fit_rate_model(big_df)
        coef = model.coefficients.set_index("term")["multiplier"]
        assert coef["Exam day"] == pytest.approx(EXAM_EFFECT, abs=0.04)
        assert coef["Friday"] == pytest.approx(WEEKDAY_EFFECT["Friday"] / WEEKDAY_EFFECT["Monday"], abs=0.04)
        assert 0 < model.r_squared < 1

    def test_predicted_means_match_data(self, big_df):
        model = fc.fit_rate_model(big_df)
        regular_mondays = big_df[(big_df["day"] == "Monday") & (big_df["day_type"] == "Regular")]
        observed = regular_mondays.groupby("time_interval")["arrivals"].mean()
        predicted = model.predict("Monday")
        assert np.allclose(predicted[observed.index], observed, rtol=0.06)
        assert (model.predict("Monday", "Exam") < predicted).all()

    def test_unseen_levels_raise(self, big_df):
        model = fc.fit_rate_model(big_df)
        with pytest.raises(ValueError):
            model.predict("Sunday")
        with pytest.raises(ValueError):
            model.predict("Monday", "Festival")

    def test_day_effect_tests(self, big_df):
        out = fc.day_effect_tests(big_df)
        assert out["day_type"]["t"] < 0 and out["day_type"]["t_p"] < 0.05  # exam days are quieter
        assert 0 <= out["weekday"]["anova_p"] <= 1 and 0 <= out["weekday"]["kruskal_p"] <= 1


class TestOverdispersion:
    def test_variance_near_truth(self, big_df):
        # generator: day factor CV 5% -> Var(W) = 0.0025
        v = fc.estimate_day_variance(big_df, fc.fit_rate_model(big_df))["var_w"]
        assert 0.0008 < v < 0.005

    def test_pure_poisson_not_rejected(self):
        df = poisson_only_df()
        out = fc.estimate_day_variance(df, fc.fit_rate_model(df))
        assert out["var_w"] < 0.003
        assert out["p_value"] > 0.01


class TestPeakAndUpdate:
    def test_peak_probabilities_sum_to_one(self):
        mu = pd.Series([10.0, 30.0, 29.0, 12.0], index=list("abcd"))
        out = fc.peak_time_distribution(mu, 30, 0.002, n_sims=5000, seed=1)
        assert out["p_peak"].sum() == pytest.approx(1.0)
        assert out["most_likely_peak"] == "b"
        assert set(out["top_two"]) == {"b", "c"}
        assert 0 <= out["p_any_overload"] <= 1

    def test_dominant_slot_is_almost_surely_the_peak(self):
        mu = pd.Series([5.0, 60.0, 5.0], index=list("xyz"))
        assert fc.peak_time_distribution(mu, 100, 0.0, n_sims=2000)["p_peak"]["y"] > 0.999

    def test_posterior_formula(self):
        shape, rate = fc.update_day_factor([20, 30], [25, 40], var_w=0.01)
        assert shape == pytest.approx(100 + 65)
        assert rate == pytest.approx(100 + 50)

    def test_busy_morning_raises_later_risk(self):
        mu = pd.Series([20.0, 25.0, 30.0, 28.0], index=["s1", "s2", "s3", "s4"])
        busy = fc.same_day_update(mu, pd.Series([28, 34], index=["s1", "s2"]), 30, 0.01)
        quiet = fc.same_day_update(mu, pd.Series([14, 18], index=["s1", "s2"]), 30, 0.01)
        assert list(busy["period"]) == ["s3", "s4"]
        assert (busy["p_after"] > busy["p_before"]).all()
        assert (quiet["p_after"] < quiet["p_before"]).all()


class TestSmallTests:
    def test_runs_test(self):
        alternating = fc.runs_test([True, False] * 10)
        assert alternating["runs"] == 20 and alternating["z"] > 0
        clustered = fc.runs_test([True] * 10 + [False] * 10)
        assert clustered["runs"] == 2 and clustered["p_value"] < 0.01
        assert np.isnan(fc.runs_test([True] * 5)["z"])

    def test_proportion_z_test(self):
        out = fc.proportion_z_test(60, 100, 0.5)
        assert out["z"] == pytest.approx(2.0)
        assert out["p_value"] == pytest.approx(2 * stats.norm.sf(2.0))
        assert fc.proportion_z_test(60, 100, 0.5, "greater")["p_value"] == pytest.approx(stats.norm.sf(2.0))

    def test_brier_score(self):
        assert fc.brier_score([1.0, 0.0], [1, 0]) == 0.0
        assert fc.brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)

    def test_consecutive_correlation_shape(self, semester_df):
        model = fc.fit_rate_model(semester_df)
        table = fc.consecutive_slot_correlation(semester_df, model)
        assert len(table) == semester_df["time_interval"].nunique() - 1
        assert table["rho"].between(-1, 1).all()


class TestBacktest:
    def test_no_future_leakage(self, semester_df):
        bt1 = fc.rolling_backtest(semester_df, 30)
        dates = sorted(semester_df["date"].unique())
        changed = semester_df.copy()
        late = changed["date"] >= dates[70]
        changed.loc[late, "arrivals"] = changed.loc[late, "arrivals"] * 3  # alter only the future
        bt2 = fc.rolling_backtest(changed, 30)
        early = lambda b: b.slots[b.slots["date"] < dates[70]].reset_index(drop=True)
        pd.testing.assert_frame_equal(early(bt1)[["mu_model", "p_model"]], early(bt2)[["mu_model", "p_model"]])

    def test_summary(self, semester_df):
        bt = fc.rolling_backtest(semester_df, 30)
        assert len(bt.days) == semester_df["date"].nunique() - 30  # unseen exam type falls back, no day skipped
        out = fc.summarize_backtest(bt)
        assert len(out["tests"]) == 6
        assert out["tests"]["p_value"].between(0, 1).all()
        assert 0.85 < out["coverage"] <= 1.0
        assert out["hit_rate_model"] > out["hit_chance"]
        assert sum(out["pit_counts"]) == len(bt.slots)

    def test_requires_enough_days(self, semester_df):
        with pytest.raises(ValueError):
            fc.rolling_backtest(semester_df, 30, train_days=500)


def test_cleaning_fills_and_normalises_day_type():
    raw = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "time_interval": ["12:00-12:15"] * 2,
                        "arrivals": [5, 6]})
    df, _ = clean_dataset(raw)
    assert (df["day_type"] == "Regular").all()
    df2, _ = clean_dataset(raw.assign(day_type=["exam", None]))
    assert list(df2["day_type"]) == ["Exam", "Regular"]
