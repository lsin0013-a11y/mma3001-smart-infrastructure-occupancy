"""Tests for smart_infra.forecasting."""

import numpy as np
import pandas as pd
import pytest

from smart_infra.forecasting import (
    LAG_COLUMNS,
    TARGET,
    PersistenceBaseline,
    TimeOfWeekBaseline,
    evaluate,
    make_dataset,
    make_models,
    split_by_date,
)

TZ = "Australia/Melbourne"


def make_table(days=10, freq="5min", seed=0):
    """Synthetic occupancy table: busy on weekday afternoons, plus noise."""
    index = pd.date_range("2024-05-06", periods=days * pd.Timedelta("1D") // pd.Timedelta(freq), freq=freq, tz=TZ)
    rng = np.random.default_rng(seed)
    busy = (index.dayofweek < 5) & (index.hour >= 12) & (index.hour < 17)
    headcount = np.where(busy, 4.0, 0.0) + rng.uniform(0, 0.5, len(index))
    return pd.DataFrame({"mean_headcount": headcount, "outage": False}, index=index)


def test_make_dataset_only_uses_finished_intervals():
    table = make_table()
    data = make_dataset(table)
    # At 5-min resolution the latest interval known 10 min ahead is 2 intervals back.
    t = data.index[100]
    assert data.loc[t, "recent"] == table.loc[t - pd.Timedelta("10min"), TARGET]
    assert data.loc[t, "same_time_last_week"] == table.loc[t - pd.Timedelta("7D"), TARGET]
    hour = table[TARGET].loc[t - pd.Timedelta("65min"): t - pd.Timedelta("10min")]
    assert data.loc[t, "recent_hour_mean"] == pytest.approx(hour.mean())


def test_make_dataset_never_uses_outage_values():
    table = make_table()
    out = table.index[2500]  # after the first week, so it would be a target
    table.loc[out, "outage"] = True
    table.loc[out, TARGET] = 999.0  # a value that must never leak into inputs

    data = make_dataset(table)

    assert out not in data.index
    assert (data[LAG_COLUMNS] < 999).all().all()


def test_make_dataset_rejects_irregular_index():
    table = make_table().drop(make_table().index[5])
    with pytest.raises(ValueError):
        make_dataset(table)


def test_split_by_date_is_chronological():
    data = make_dataset(make_table(days=14))
    train, test = split_by_date(data, test_start="2024-05-17")
    assert train.index.max() < test.index.min()


def test_baselines():
    data = make_dataset(make_table(days=14))
    X, y = data.drop(columns=TARGET), data[TARGET]

    assert np.array_equal(PersistenceBaseline().fit(X, y).predict(X), X["recent"].to_numpy())
    profile = TimeOfWeekBaseline().fit(X, y).predict(X)
    busy = (X["day_of_week"] < 5) & (X["minute_of_day"] >= 12 * 60) & (X["minute_of_day"] < 17 * 60)
    assert profile[busy.to_numpy()].mean() > 4 > profile[~busy.to_numpy()].mean()


def test_models_learn_the_pattern_and_q90_sits_above_the_mean():
    data = make_dataset(make_table(days=21))
    train, test = split_by_date(data, test_start="2024-05-24")
    models = make_models({"max_iter": 50})
    preds = {}
    for name in ("linear", "boosting", "boosting_q90"):
        model = models[name].fit(train.drop(columns=TARGET), train[TARGET])
        preds[name] = model.predict(test.drop(columns=TARGET))
        assert (preds[name] >= 0).all()
    assert evaluate(test[TARGET], preds["boosting"])["mae"] < 0.5
    assert evaluate(test[TARGET], preds["boosting_q90"])["under_share"] < 0.3


def test_evaluate_metrics():
    y = pd.Series([1.0, 2.0, 3.0])
    result = evaluate(y, np.array([2.0, 2.0, 2.0]), quantile=0.9)
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["bias"] == pytest.approx(0.0)
    assert result["under_share"] == pytest.approx(1 / 3)
    # Pinball at q=0.9: under-forecast by 1 costs 0.9, over-forecast by 1 costs 0.1.
    assert result["pinball"] == pytest.approx((0.1 + 0 + 0.9) / 3)
