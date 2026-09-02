"""Forecaster tests. These use synthetic series, so they run offline and fast."""

from __future__ import annotations

import numpy as np
import pytest

from kuntadata.forecast import MIN_POINTS, forecast
from kuntadata.statfin import Series


def make_series(values: list[float], start_year: int = 1990) -> Series:
    return Series(
        area="Testilä",
        indicator="Väkiluku",
        years=list(range(start_year, start_year + len(values))),
        values=values,
    )


def test_follows_a_clean_linear_trend():
    series = make_series([100.0 + 10 * i for i in range(20)])
    result = forecast(series, horizon=3)

    assert result.years == [2010, 2011, 2012]
    # A perfectly linear series should be projected almost exactly.
    assert result.values[0] == pytest.approx(300.0, rel=0.02)
    assert result.values[-1] == pytest.approx(320.0, rel=0.02)


def test_reports_a_backtest_against_the_naive_baseline():
    series = make_series([100.0 + 10 * i for i in range(20)])
    backtest = forecast(series, horizon=3).backtest

    assert backtest is not None
    assert backtest.folds > 0
    assert backtest.mape >= 0
    assert backtest.baseline_mape >= 0


def test_declining_series_is_projected_downward():
    series = make_series([5000.0 - 60 * i for i in range(25)])
    result = forecast(series, horizon=5)
    assert result.values[-1] < series.values[-1]


def test_refuses_a_series_too_short_to_validate():
    series = make_series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="at least"):
        forecast(series, horizon=2)


def test_minimum_length_is_actually_sufficient():
    series = make_series([100.0 + i for i in range(MIN_POINTS)])
    result = forecast(series, horizon=1)
    assert len(result.values) == 1


def test_rejects_a_nonsense_horizon():
    series = make_series([100.0 + i for i in range(15)])
    with pytest.raises(ValueError, match="horizon"):
        forecast(series, horizon=0)


def test_noise_does_not_beat_the_baseline_by_much():
    """On a pure random walk there is no trend to find.

    The model should not claim large skill here. This guards against a change
    that makes the backtest flatter itself.
    """
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 5, 40)) + 1000
    result = forecast(make_series([float(v) for v in walk]), horizon=3)

    assert result.backtest is not None
    assert result.backtest.skill < 0.5
