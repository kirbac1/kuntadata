"""
Time-series forecasting for municipal indicators.

These are short annual series — at most 39 points, often fewer. That rules out
most of the usual toolbox: ARIMA order selection on 39 observations mostly
fits noise, and anything seasonal is meaningless on annual data with no
within-year cycle.

What suits the data is Holt's linear trend (double exponential smoothing): two
parameters, a level and a trend, both of which mean something for a population
or an employment rate. It is fitted here by a small grid search on
walk-forward error rather than on in-sample fit, because in-sample fit on 39
points is not evidence of anything.

Every forecast is returned alongside the error of a naive drift baseline over
the same backtest. If the model cannot beat "assume the last change repeats",
the caller can see that immediately — a forecast without that comparison is
decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .statfin import Series


@dataclass(frozen=True)
class Backtest:
    """Walk-forward error, and the same error for a naive baseline."""

    mape: float
    baseline_mape: float
    folds: int

    @property
    def beats_baseline(self) -> bool:
        return self.mape < self.baseline_mape

    @property
    def skill(self) -> float:
        """Fraction of the baseline's error removed. Negative means worse."""
        if self.baseline_mape == 0:
            return 0.0
        return 1.0 - (self.mape / self.baseline_mape)


@dataclass(frozen=True)
class Forecast:
    area: str
    indicator: str
    years: list[int]
    values: list[float]
    backtest: Backtest | None
    method: str = "holt-linear"
    params: dict[str, float] = field(default_factory=dict)

    def as_records(self) -> list[dict[str, float | int]]:
        return [{"year": y, "value": v} for y, v in zip(self.years, self.values, strict=True)]


def _holt(values: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
    """Run Holt's linear trend over a series, returning final (level, trend)."""
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    for observation in values[1:]:
        previous_level = level
        level = alpha * float(observation) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
    return level, trend


def _project(level: float, trend: float, horizon: int) -> list[float]:
    return [level + trend * step for step in range(1, horizon + 1)]


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring zero actuals."""
    mask = actual != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# Deliberately coarse: a fine grid on 39 points is false precision.
_ALPHAS = (0.2, 0.4, 0.6, 0.8)
_BETAS = (0.05, 0.1, 0.3, 0.5)

MIN_POINTS = 8


def _walk_forward(
    values: np.ndarray, alpha: float, beta: float, horizon: int, min_train: int
) -> tuple[float, float, int]:
    """Expanding-window backtest. Returns (model MAPE, baseline MAPE, folds)."""
    model_errors: list[float] = []
    baseline_errors: list[float] = []

    for cutoff in range(min_train, len(values) - horizon + 1):
        train, actual = values[:cutoff], values[cutoff : cutoff + horizon]

        level, trend = _holt(train, alpha, beta)
        model_errors.append(_mape(actual, np.array(_project(level, trend, horizon))))

        # Naive drift: keep repeating the most recent year-on-year change.
        step = float(train[-1] - train[-2]) if len(train) > 1 else 0.0
        drift = np.array([train[-1] + step * h for h in range(1, horizon + 1)])
        baseline_errors.append(_mape(actual, drift))

    if not model_errors:
        return float("inf"), float("inf"), 0
    return float(np.mean(model_errors)), float(np.mean(baseline_errors)), len(model_errors)


def forecast(series: Series, horizon: int = 5) -> Forecast:
    """Project a series forward, selecting parameters by walk-forward error.

    Raises ValueError when the series is too short to both fit and validate —
    returning a number anyway would imply a confidence the data cannot support.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 year")
    if len(series) < MIN_POINTS:
        raise ValueError(
            f"{series.area} / {series.indicator} has {len(series)} points; "
            f"at least {MIN_POINTS} are needed to fit and validate a forecast"
        )

    values = np.asarray(series.values, dtype=float)
    min_train = max(4, len(values) // 2)

    best: tuple[float, float, float, float, int] | None = None
    for alpha in _ALPHAS:
        for beta in _BETAS:
            model_mape, baseline_mape, folds = _walk_forward(
                values, alpha, beta, horizon, min_train
            )
            if folds and (best is None or model_mape < best[0]):
                best = (model_mape, baseline_mape, alpha, beta, folds)

    if best is None:
        raise ValueError(
            f"{series.area} / {series.indicator}: not enough history to backtest a "
            f"{horizon}-year horizon"
        )

    model_mape, baseline_mape, alpha, beta, folds = best
    level, trend = _holt(values, alpha, beta)
    projected = _project(level, trend, horizon)
    last_year = series.years[-1]

    return Forecast(
        area=series.area,
        indicator=series.indicator,
        years=[last_year + step for step in range(1, horizon + 1)],
        values=projected,
        backtest=Backtest(mape=model_mape, baseline_mape=baseline_mape, folds=folds),
        params={"alpha": alpha, "beta": beta},
    )
