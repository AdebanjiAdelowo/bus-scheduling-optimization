import numpy as np
import pandas as pd
import pytest

from busopt.forecast.evaluate import metrics
from busopt.forecast.features import make_features
from busopt.forecast.models import GBTModel, HistoricalAverage, RidgeModel, RollingAverage, SeasonalNaive
from busopt.synthetic import toy_panel, toy_weather

TRAIN_END = "2025-03-31"


@pytest.fixture(scope="module")
def panel():
    return toy_panel()


@pytest.mark.parametrize("h", [1, 7, 14])
def test_features_do_not_use_data_after_origin(panel, h):
    """Changing demand on or after day D - h + 1 must not change any feature of day D."""
    w = toy_weather(panel)
    base = make_features(panel, w, h, TRAIN_END)
    target = panel.ts.max().normalize()
    origin_end = target - pd.Timedelta(days=h) + pd.Timedelta(hours=23)
    tampered = panel.copy()
    tampered.loc[tampered.ts > origin_end, "ridership"] += 10_000
    alt = make_features(tampered, w, h, TRAIN_END)
    cols = [c for c in base.columns if c != "y"]
    m = base.ts >= target
    pd.testing.assert_frame_equal(base.loc[m, cols], alt.loc[m, cols])
    assert not base.loc[m, "y"].equals(alt.loc[m, "y"])


def test_seasonal_naive_is_last_admissible_week(panel):
    f = make_features(panel, toy_weather(panel), 7, TRAIN_END)
    row = f[(f.route_id == "T1") & (f.ts == pd.Timestamp("2025-04-20 08:00"))].iloc[0]
    truth = panel[(panel.route_id == "T1") & (panel.ts == pd.Timestamp("2025-04-13 08:00"))].ridership.iloc[0]
    assert SeasonalNaive().predict(pd.DataFrame([row]))[0] == truth
    f14 = make_features(panel, toy_weather(panel), 14, TRAIN_END)
    row = f14[(f14.route_id == "T1") & (f14.ts == pd.Timestamp("2025-04-20 08:00"))].iloc[0]
    truth = panel[(panel.route_id == "T1") & (panel.ts == pd.Timestamp("2025-04-06 08:00"))].ridership.iloc[0]
    assert row.lag_w0 == truth


def test_models_fit_and_beat_nothing_silly(panel):
    f = make_features(panel, toy_weather(panel), 1, TRAIN_END)
    f = f[f.y.notna() & f.mean_w4.notna()]
    tr, va, te = (
        f[f.ts < "2025-03-15"],
        f[(f.ts >= "2025-03-15") & (f.ts < "2025-04-01")],
        f[f.ts >= "2025-04-01"],
    )
    scores = {}
    for m in (SeasonalNaive(), HistoricalAverage(), RollingAverage(), RidgeModel(), GBTModel(max_iter=60)):
        p = m.fit(tr, va).predict(te)
        assert p.shape == (len(te),) and np.isfinite(p).all() and (p >= 0).all()
        scores[m.name] = metrics(te.y.to_numpy(), p)["mae"]
    # on a stationary Poisson panel averaging must beat a single noisy lag
    assert scores["rolling_average_4w"] < scores["seasonal_naive"]
    assert scores["gbt"] < scores["seasonal_naive"]


def test_metrics():
    m = metrics(np.array([10.0, 20.0]), np.array([12.0, 16.0]))
    assert (
        m["mae"] == 3.0
        and np.isclose(m["rmse"], np.sqrt(10.0))
        and np.isclose(m["wape"], 0.2)
        and m["bias"] == -1.0
    )
