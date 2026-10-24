"""Short-term forecast of a room's average headcount, for ventilation control.

The forecast is issued at time *t* for the interval starting at *t + horizon*
(30 min by default). It may only use what is known at *t*: the calendar for
the target interval, and headcounts from intervals that had already finished
by *t*. :func:`make_dataset` builds those inputs, so no model can see the
future it is predicting.

Models (see :func:`make_models`):

- **time-of-week baseline**: mean headcount for that weekday and time of day
  in the training data;
- **persistence baseline**: the latest finished interval's headcount;
- **linear**: ridge regression on calendar and recent-headcount inputs;
- **gradient boosting**: trees on the same inputs, which can capture
  interactions (e.g. Friday afternoons in exam periods);
- **gradient boosting, calendar only**: the same model without recent
  headcounts, i.e. what can be forecast a day or more ahead;
- **gradient boosting, 90th percentile**: predicts a headcount that is
  exceeded only 10% of the time, because under-ventilating is worse than
  over-ventilating.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from smart_infra.academic_calendar import calendar_features

TARGET = "mean_headcount"
CALENDAR_COLUMNS = ["minute_of_day", "day_of_week", "weekend", "public_holiday", "period"]
LAG_COLUMNS = ["recent", "recent_hour_mean", "same_time_yesterday", "same_time_last_week"]


def make_dataset(table: pd.DataFrame, horizon: str = "30min", calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build forecasting inputs and target from an occupancy table.

    Parameters
    ----------
    table : pandas.DataFrame
        Occupancy table at a regular resolution (output of
        :func:`smart_infra.occupancy.build_occupancy_table`), with columns
        ``mean_headcount`` and ``outage``.
    horizon : str, default "30min"
        How far ahead the forecast is issued. Rounded up to a whole number of
        intervals, and at least one interval.
    calendar : pandas.DataFrame, optional
        Output of :func:`smart_infra.academic_calendar.calendar_features` for
        ``table.index``. Computed if ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per interval whose target and inputs are all known, with the
        :data:`CALENDAR_COLUMNS`, the :data:`LAG_COLUMNS` and the target
        ``mean_headcount``:

        - ``recent``: headcount of the latest interval finished by the time
          the forecast is issued;
        - ``recent_hour_mean``: mean headcount over the hour up to then;
        - ``same_time_yesterday`` / ``same_time_last_week``: headcount 24 h /
          7 days before the target interval.

        Outage intervals are treated as unknown: they are dropped as targets
        and never used as inputs.

    Raises
    ------
    ValueError
        If the table's index is not regular.
    """
    step = table.index[1] - table.index[0]
    if not (table.index[1:] - table.index[:-1] == step).all():
        raise ValueError("table must have a regular index")
    ahead = max(1, math.ceil(pd.Timedelta(horizon) / step))
    per_hour = max(1, round(pd.Timedelta("1h") / step))
    per_day = round(pd.Timedelta("1D") / step)

    # Shifts count intervals, and the index is regular in UTC, so "yesterday"
    # is exactly 24 h earlier (off by an hour on the two daylight-saving days).
    y = table[TARGET].where(~table["outage"])
    known = y.shift(ahead)
    lags = pd.DataFrame({
        "recent": known,
        "recent_hour_mean": known.rolling(per_hour, min_periods=per_hour).mean(),
        "same_time_yesterday": y.shift(max(per_day, ahead)),
        "same_time_last_week": y.shift(7 * per_day),
    })
    calendar = calendar_features(table.index) if calendar is None else calendar
    data = pd.concat([calendar[CALENDAR_COLUMNS], lags, y.rename(TARGET)], axis=1)
    return data.dropna(subset=LAG_COLUMNS + [TARGET])


def split_by_date(data: pd.DataFrame, test_start: str = "2026-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: everything before ``test_start`` (local date) trains.

    A random split would let the model learn from intervals just after the
    ones it is tested on, which flatters a time-series forecast.
    """
    cut = pd.Timestamp(test_start, tz=data.index.tz)
    return data[data.index < cut], data[data.index >= cut]


class TimeOfWeekBaseline:
    """Predict the training mean for the same weekday and time of day."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TimeOfWeekBaseline":
        self.profile_ = y.groupby([X["day_of_week"], X["minute_of_day"]]).mean()
        self.default_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([X["day_of_week"], X["minute_of_day"]])
        return self.profile_.reindex(keys).fillna(self.default_).to_numpy()


class PersistenceBaseline:
    """Predict that the room stays as it was in the latest finished interval."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["recent"].to_numpy()


def _encoded(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric version of the inputs for scikit-learn (period as integer codes)."""
    X = X.copy()
    X["period"] = X["period"].cat.codes
    X["weekend"] = X["weekend"].astype(int)
    X["public_holiday"] = X["public_holiday"].astype(int)
    X["hour"] = X["minute_of_day"] // 60
    return X


class SklearnModel:
    """Wrap a scikit-learn estimator so it accepts the dataset columns directly."""

    def __init__(self, estimator, columns: list[str]):
        self.estimator = estimator
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        self.estimator.fit(_encoded(X)[self.columns], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Headcount can't be negative.
        return np.clip(self.estimator.predict(_encoded(X)[self.columns]), 0, None)


def _linear() -> SklearnModel:
    categorical = ["hour", "day_of_week", "period"]
    numeric = ["weekend", "public_holiday", *LAG_COLUMNS]
    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("scale", StandardScaler(), numeric),
    ])
    return SklearnModel(make_pipeline(pre, Ridge(alpha=1.0)), categorical + numeric)


def _boosting(columns: list[str], quantile: float | None = None, **params) -> SklearnModel:
    categorical = [c in ("day_of_week", "period") for c in columns]
    loss = {"loss": "quantile", "quantile": quantile} if quantile is not None else {}
    estimator = HistGradientBoostingRegressor(categorical_features=categorical, random_state=0, **loss, **params)
    return SklearnModel(estimator, columns)


def make_models(boosting_params: dict | None = None) -> dict[str, object]:
    """Create the unfitted models compared in the study, keyed by name.

    Parameters
    ----------
    boosting_params : dict, optional
        Extra ``HistGradientBoostingRegressor`` settings (e.g. from
        :func:`tune_boosting`) applied to all gradient-boosting models.
    """
    params = boosting_params or {}
    calendar_cols = ["minute_of_day", "day_of_week", "public_holiday", "period"]
    all_cols = calendar_cols + LAG_COLUMNS
    return {
        "time_of_week_baseline": TimeOfWeekBaseline(),
        "persistence_baseline": PersistenceBaseline(),
        "linear": _linear(),
        "boosting": _boosting(all_cols, **params),
        "boosting_calendar_only": _boosting(calendar_cols, **params),
        "boosting_q90": _boosting(all_cols, quantile=0.9, **params),
    }


def tune_boosting(train: pd.DataFrame, grid: list[dict], n_splits: int = 3) -> tuple[dict, pd.DataFrame]:
    """Choose gradient-boosting settings by time-series cross-validation.

    The training period is cut into ``n_splits + 1`` consecutive blocks; each
    fold trains on the blocks before a validation block. Only training data is
    used, so the test period stays unseen.

    Parameters
    ----------
    train : pandas.DataFrame
        Training part of :func:`make_dataset`'s output.
    grid : list of dict
        Candidate ``HistGradientBoostingRegressor`` settings.
    n_splits : int, default 3
        Number of validation folds.

    Returns
    -------
    best : dict
        The settings with the lowest mean validation MAE.
    scores : pandas.DataFrame
        Mean and per-fold validation MAE for every candidate.
    """
    blocks = np.array_split(np.arange(len(train)), n_splits + 1)
    X, y = train.drop(columns=TARGET), train[TARGET]
    rows = []
    for params in grid:
        fold_mae = []
        for k in range(1, n_splits + 1):
            fit_idx, val_idx = np.concatenate(blocks[:k]), blocks[k]
            model = make_models(params)["boosting"].fit(X.iloc[fit_idx], y.iloc[fit_idx])
            fold_mae.append(np.abs(model.predict(X.iloc[val_idx]) - y.iloc[val_idx]).mean())
        rows.append({**params, "mean_mae": float(np.mean(fold_mae)), "fold_mae": [round(m, 4) for m in fold_mae]})
    best = grid[int(np.argmin([r["mean_mae"] for r in rows]))]
    return best, pd.DataFrame(rows).sort_values("mean_mae", ignore_index=True)


def evaluate(y_true: pd.Series, y_pred: np.ndarray, quantile: float | None = None) -> dict[str, float]:
    """Forecast accuracy metrics.

    Parameters
    ----------
    y_true : pandas.Series
        Observed mean headcount.
    y_pred : numpy.ndarray
        Forecast mean headcount.
    quantile : float, optional
        If the forecast is a quantile, also report the pinball loss for it.

    Returns
    -------
    dict
        ``mae`` and ``rmse`` (people), ``bias`` (mean of forecast - observed),
        ``under_share`` (share of intervals where the forecast is below the
        observation, i.e. the room would be under-ventilated), and
        ``pinball`` if ``quantile`` is given.
    """
    err = np.asarray(y_pred) - y_true.to_numpy()
    metrics = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "under_share": float((err < -1e-9).mean()),
    }
    if quantile is not None:
        metrics["pinball"] = float(np.mean(np.maximum(quantile * -err, (quantile - 1) * -err)))
    return metrics
