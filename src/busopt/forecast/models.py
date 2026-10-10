"""Forecasting models with a common ``fit(train) / predict(frame)`` interface.

Every model consumes the feature frame from :mod:`busopt.forecast.features`, whose demand-derived
columns are already restricted to information available at the forecast origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from busopt.forecast.features import CALENDAR_FEATURES, DEMAND_FEATURES, WEATHER_COLS


class SeasonalNaive:
    """Demand at the same hour on the most recent admissible same weekday."""

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> SeasonalNaive:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # fall back to older weeks only when the lag day itself is missing (reporting outage)
        return df.lag_w0.fillna(df.lag_w1).fillna(df.lag_w2).fillna(df.train_profile).to_numpy()


class HistoricalAverage:
    """Training-window mean by route, day type and hour. Static: never updated after training."""

    name = "historical_average"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> HistoricalAverage:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df.train_profile.to_numpy()


class RollingAverage:
    """Mean of the four most recent admissible same-weekday observations."""

    name = "rolling_average_4w"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> RollingAverage:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df.mean_w4.fillna(df.train_profile).to_numpy()


def _design(df: pd.DataFrame, weather: bool) -> pd.DataFrame:
    """Numeric design matrix for the linear model: demand features plus hour x day-type dummies."""
    x = df[DEMAND_FEATURES].copy()
    for c in DEMAND_FEATURES:
        x[c] = x[c].fillna(df.train_profile)
    x["holiday"] = df.holiday.to_numpy()
    x["holiday_w0"] = df.holiday_w0.to_numpy()
    x["doy_sin"], x["doy_cos"] = df.doy_sin.to_numpy(), df.doy_cos.to_numpy()
    cell = (df.daytype * 24 + df.hour).to_numpy()
    dummies = np.zeros((len(df), 72), dtype=np.float32)
    dummies[np.arange(len(df)), cell] = 1.0
    x = pd.concat(
        [x.reset_index(drop=True), pd.DataFrame(dummies, columns=[f"c{i}" for i in range(72)])], axis=1
    )
    if weather:
        for c in WEATHER_COLS:
            x[c] = df[c].to_numpy()
    return x


class RidgeModel:
    """One ridge regression per route on lagged demand, calendar dummies and optional weather."""

    def __init__(self, weather: bool = False, alpha: float = 10.0):
        self.weather, self.alpha = weather, alpha
        self.name = "ridge" + ("_weather_oracle" if weather else "")
        self.models: dict[str, object] = {}

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> RidgeModel:
        for route, sub in train.groupby("route_id"):
            self.models[route] = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(
                _design(sub, self.weather), sub.y
            )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        pos = np.arange(len(df))
        for route, idx in df.groupby("route_id").indices.items():
            out[pos[idx]] = self.models[route].predict(_design(df.iloc[idx], self.weather))
        return np.clip(out, 0.0, None)


class GBTModel:
    """Global histogram gradient-boosted trees with Poisson loss and route as a categorical feature.

    The number of boosting rounds is chosen on the temporal validation split, then the model is
    refit on train + validation with that fixed number of rounds.
    """

    def __init__(
        self, weather: bool = False, seed: int = 17, max_iter: int = 500, learning_rate: float = 0.06
    ):
        self.weather, self.seed, self.max_iter, self.lr = weather, seed, max_iter, learning_rate
        self.name = "gbt" + ("_weather_oracle" if weather else "")
        self.best_iter: int | None = None

    def _cols(self) -> list[str]:
        return ["route_code", *DEMAND_FEATURES, *CALENDAR_FEATURES, *(WEATHER_COLS if self.weather else [])]

    def _make(self, n_iter: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=n_iter,
            learning_rate=self.lr,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=[0],
            early_stopping=False,
            random_state=self.seed,
        )

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> GBTModel:
        cols = self._cols()
        if valid is not None and len(valid):
            m = self._make(self.max_iter).fit(train[cols], train.y)
            errs = [np.mean(np.abs(p - valid.y.to_numpy())) for p in m.staged_predict(valid[cols])]
            self.best_iter = int(np.argmin(errs)) + 1
            # out-of-sample validation forecasts of the train-only model: the residual pool for scenarios
            self.valid_pred = self._make(self.best_iter).fit(train[cols], train.y).predict(valid[cols])
            self.valid_mae_curve = errs
            full = pd.concat([train, valid])
        else:
            self.best_iter, full = self.max_iter, train
        self.model = self._make(self.best_iter).fit(full[cols], full.y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self._cols()])
