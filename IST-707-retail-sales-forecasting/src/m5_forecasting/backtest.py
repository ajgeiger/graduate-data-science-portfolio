"""Rolling-origin backtest for the M5 sales forecast.

The original notebook validated on rows sampled at random, and some features
(item sales mean and std, the price counts) used the full history, so future
days leaked into training. Here the same features from features.py are built
only from data up to each forecast origin. The model trains on the year
before the origin and forecasts the next 28 days one day at a time, feeding
its predictions back in as the original notebook did.

Data is kept in numpy arrays shaped (series, day) so a full run fits in a
few GB of memory.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

FORECAST_HORIZON = 28
LAGS = [7, 28]
WINDOWS = [7, 28]
ID_COLUMNS = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
CALENDAR_COLUMNS = [
    "wday",
    "month",
    "year",
    *EVENT_COLUMNS,
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "week",
    "quarter",
    "mday",
]

# same settings as the original forecast notebook
LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "metric": "rmse",
    "force_row_wise": True,
    "learning_rate": 0.075,
    "sub_feature": 0.82,
    "sub_row": 0.78,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "num_leaves": 128,
    "min_data_in_leaf": 104,
    "verbosity": -1,
    "seed": 0,
}
NUM_ROUNDS = 1250

FEATURES = [
    *ID_COLUMNS,
    "wday",
    "month",
    "year",
    *EVENT_COLUMNS,
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rmean_{lag}_{window}" for window in WINDOWS for lag in LAGS],
    "week",
    "quarter",
    "mday",
    "price_nunique_items",
    "item_nunique_price",
    "item_nunique_weekly",
    "enc_item_id_mean",
    "enc_item_id_std",
]
CATEGORICAL_FEATURES = [*ID_COLUMNS, *EVENT_COLUMNS]
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}


class M5Data:
    """The M5 files as arrays indexed by (series, day)."""

    def __init__(self, path: str):
        sales = pd.read_csv(f"{path}/sales_train_evaluation.csv")
        day_columns = [column for column in sales.columns if column.startswith("d_")]
        n_days = len(day_columns)
        self.series_ids = sales["id"].values

        self.id_codes = {}
        self.id_labels = {}
        for column in ID_COLUMNS:
            codes = sales[column].astype("category").cat.codes
            self.id_codes[column] = codes.values.astype(np.int16)
            self.id_labels[column] = sales[column].values

        self.units_sold = sales[day_columns].values.astype(np.float32)
        del sales
        gc.collect()

        calendar = pd.read_csv(f"{path}/calendar.csv").iloc[: n_days + FORECAST_HORIZON].copy()
        dates = pd.to_datetime(calendar["date"])
        self.calendar = pd.DataFrame(
            {
                "wday": calendar["wday"].astype(np.int16),
                "month": calendar["month"].astype(np.int16),
                "year": calendar["year"].astype(np.int16),
                "week": dates.dt.isocalendar().week.astype(np.int16).values,
                "quarter": dates.dt.quarter.astype(np.int16),
                "mday": dates.dt.day.astype(np.int16),
                "snap_CA": calendar["snap_CA"].astype(np.float32),
                "snap_TX": calendar["snap_TX"].astype(np.float32),
                "snap_WI": calendar["snap_WI"].astype(np.float32),
            }
        )
        for column in EVENT_COLUMNS:
            # shift by one so "no event" is 0 instead of -1
            codes = calendar[column].astype("category").cat.codes + 1
            self.calendar[column] = codes.astype(np.int16)

        self.sell_price = self._daily_prices(path, calendar["wm_yr_wk"].values)
        gc.collect()

    def _daily_prices(self, path: str, day_weeks: np.ndarray) -> np.ndarray:
        """Weekly prices spread to days. NaN means the item was not on sale."""
        prices = pd.read_csv(f"{path}/sell_prices.csv")
        weeks = np.sort(prices["wm_yr_wk"].unique())
        week_index = {week: index for index, week in enumerate(weeks)}

        series_index = pd.Series(
            np.arange(len(self.series_ids)),
            index=pd.MultiIndex.from_arrays(
                [self.id_labels["store_id"], self.id_labels["item_id"]]
            ),
        )
        price_rows = series_index.reindex(
            pd.MultiIndex.from_arrays([prices["store_id"].values, prices["item_id"].values])
        ).values
        price_columns = prices["wm_yr_wk"].map(week_index).values

        weekly_prices = np.full((len(self.series_ids), len(weeks)), np.nan, dtype=np.float32)
        weekly_prices[price_rows, price_columns] = prices["sell_price"].values.astype(np.float32)
        day_to_week = np.array([week_index[week] for week in day_weeks])
        return weekly_prices[:, day_to_week]


@dataclass
class WindowStats:
    """Item and price features fit on the training window only."""

    store_codes: np.ndarray
    price_nunique_items: np.ndarray
    items_per_store_price: pd.Series
    item_nunique_weekly: np.ndarray
    item_sales_mean: np.ndarray
    item_sales_std: np.ndarray

    def item_nunique_price(self, day_prices: np.ndarray) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([self.store_codes, day_prices])
        return self.items_per_store_price.reindex(keys).values.astype(np.float32)


def fit_window_stats(data: M5Data, origin_index: int, train_days: int) -> WindowStats:
    """Same features as add_price_and_item_features in features.py, computed
    from the train_days days ending at origin_index (0-based, inclusive)."""
    start = origin_index - train_days + 1
    units_sold = data.units_sold[:, start : origin_index + 1]
    prices = data.sell_price[:, start : origin_index + 1]
    on_sale = ~np.isnan(prices)
    store_codes = data.id_codes["store_id"]
    item_codes = data.id_codes["item_id"]

    price_nunique_items = np.array(
        [len(np.unique(row[~np.isnan(row)])) for row in prices], dtype=np.float32
    )

    series_rows, day_columns = np.nonzero(on_sale)
    store_prices = pd.DataFrame(
        {
            "store": store_codes[series_rows],
            "price": prices[series_rows, day_columns],
            "item": item_codes[series_rows],
        }
    )
    items_per_store_price = store_prices.groupby(["store", "price"])["item"].nunique()
    del store_prices

    weeks = data.calendar["week"].values[start : origin_index + 1]
    store_weeks = pd.DataFrame(
        {
            "store": store_codes[series_rows],
            "week": weeks[day_columns],
            "item": item_codes[series_rows],
        }
    )
    items_per_store_week = store_weeks.groupby(["store", "week"])["item"].nunique()
    del store_weeks
    by_store = np.full((store_codes.max() + 1, 54), np.nan, dtype=np.float32)
    for (store, week), count in items_per_store_week.items():
        by_store[store, week] = count
    item_nunique_weekly = by_store[store_codes]

    # item mean and std of daily sales across stores, on-sale days only
    sales_on_sale = np.where(on_sale, units_sold, np.nan)
    totals = pd.DataFrame(
        {
            "item": item_codes,
            "sum": np.nansum(sales_on_sale, axis=1),
            "sum_squares": np.nansum(sales_on_sale**2, axis=1),
            "days": on_sale.sum(axis=1),
        }
    )
    by_item = totals.groupby("item")[["sum", "sum_squares", "days"]].sum()
    mean = by_item["sum"] / by_item["days"]
    variance = (by_item["sum_squares"] - by_item["days"] * mean**2) / (by_item["days"] - 1)

    std = np.sqrt(variance.clip(lower=0))

    return WindowStats(
        store_codes=store_codes,
        price_nunique_items=price_nunique_items,
        items_per_store_price=items_per_store_price,
        item_nunique_weekly=item_nunique_weekly,
        item_sales_mean=mean.reindex(item_codes).values.astype(np.float32),
        item_sales_std=std.reindex(item_codes).values.astype(np.float32),
    )


def day_features(data: M5Data, history: np.ndarray, day: int, stats: WindowStats) -> np.ndarray:
    """Feature rows for every series on one day (0-based). history holds the
    sales before that day: actuals up to the origin, predictions after it."""
    rows = np.empty((history.shape[0], len(FEATURES)), dtype=np.float32)
    for column in ID_COLUMNS:
        rows[:, FEATURE_INDEX[column]] = data.id_codes[column]

    calendar_day = data.calendar.iloc[day]
    for column in CALENDAR_COLUMNS:
        rows[:, FEATURE_INDEX[column]] = calendar_day[column]

    day_prices = data.sell_price[:, day]
    rows[:, FEATURE_INDEX["sell_price"]] = day_prices

    # same as the pandas shift and rolling mean in features.py
    for lag in LAGS:
        rows[:, FEATURE_INDEX[f"lag_{lag}"]] = history[:, day - lag]
        for window in WINDOWS:
            lagged_window = history[:, day - lag - window + 1 : day - lag + 1]
            rows[:, FEATURE_INDEX[f"rmean_{lag}_{window}"]] = lagged_window.mean(axis=1)

    week = int(calendar_day["week"])
    rows[:, FEATURE_INDEX["price_nunique_items"]] = stats.price_nunique_items
    rows[:, FEATURE_INDEX["item_nunique_price"]] = stats.item_nunique_price(day_prices)
    rows[:, FEATURE_INDEX["item_nunique_weekly"]] = stats.item_nunique_weekly[:, week]
    rows[:, FEATURE_INDEX["enc_item_id_mean"]] = stats.item_sales_mean
    rows[:, FEATURE_INDEX["enc_item_id_std"]] = stats.item_sales_std
    return rows


def build_training_set(
    data: M5Data, origin_index: int, train_days: int, stats: WindowStats
) -> tuple[np.ndarray, np.ndarray]:
    """Features and targets for every series and day in the window where the
    item was on sale."""
    start = origin_index - train_days + 1
    on_sale = ~np.isnan(data.sell_price[:, start : origin_index + 1])
    n_rows = int(on_sale.sum())
    features = np.empty((n_rows, len(FEATURES)), dtype=np.float32)
    targets = np.empty(n_rows, dtype=np.float32)

    row = 0
    for offset, day in enumerate(range(start, origin_index + 1)):
        mask = on_sale[:, offset]
        block = day_features(data, data.units_sold, day, stats)[mask]
        features[row : row + len(block)] = block
        targets[row : row + len(block)] = data.units_sold[mask, day]
        row += len(block)
    return features, targets


def recursive_forecast(
    data: M5Data, model: lgb.Booster, origin_index: int, stats: WindowStats
) -> np.ndarray:
    """Forecast the 28 days after the origin, one day at a time."""
    empty_days = np.zeros((data.units_sold.shape[0], FORECAST_HORIZON), dtype=np.float32)
    history = np.concatenate([data.units_sold[:, : origin_index + 1], empty_days], axis=1)
    for step in range(FORECAST_HORIZON):
        day = origin_index + 1 + step
        predictions = model.predict(day_features(data, history, day, stats))
        history[:, day] = np.clip(predictions, 0, None)
    return history[:, origin_index + 1 :]


def seasonal_naive(data: M5Data, origin_index: int) -> np.ndarray:
    """Repeat the last observed week."""
    last_week = data.units_sold[:, origin_index - 6 : origin_index + 1]
    return np.tile(last_week, (1, FORECAST_HORIZON // 7))


def moving_average(data: M5Data, origin_index: int, window: int = 28) -> np.ndarray:
    """Flat forecast at the mean of the last 28 days."""
    recent = data.units_sold[:, origin_index - window + 1 : origin_index + 1]
    return np.repeat(recent.mean(axis=1, keepdims=True), FORECAST_HORIZON, axis=1)


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def wape(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(predicted - actual).sum() / actual.sum())


class WRMSSE:
    """The M5 accuracy metric. Errors are scored at 12 aggregation levels
    (42,840 series), weighted by dollar sales in the 28 days up to the origin
    and scaled by each series' in-sample one-day naive error."""

    def __init__(self, data: M5Data, origin_index: int):
        n_series = data.units_sold.shape[0]
        labels = data.id_labels
        levels = [
            np.zeros(n_series, dtype=object),
            labels["state_id"],
            labels["store_id"],
            labels["cat_id"],
            labels["dept_id"],
            join_labels(labels["state_id"], labels["cat_id"]),
            join_labels(labels["state_id"], labels["dept_id"]),
            join_labels(labels["store_id"], labels["cat_id"]),
            join_labels(labels["store_id"], labels["dept_id"]),
            labels["item_id"],
            join_labels(labels["item_id"], labels["state_id"]),
            data.series_ids,
        ]
        level_matrices = []
        for level_labels in levels:
            codes, _ = pd.factorize(level_labels)
            level_matrices.append(
                sparse.csr_matrix(
                    (np.ones(n_series, dtype=np.float32), (codes, np.arange(n_series))),
                    shape=(codes.max() + 1, n_series),
                )
            )
        self.aggregation = sparse.vstack(level_matrices).tocsr()
        level_sizes = [matrix.shape[0] for matrix in level_matrices]

        history = self.aggregation @ data.units_sold[:, : origin_index + 1]
        has_started = np.cumsum(history, axis=1) > 0
        squared_steps = np.diff(history, axis=1) ** 2
        counted = has_started[:, :-1]
        self.scale = (squared_steps * counted).sum(axis=1) / np.maximum(counted.sum(axis=1), 1)
        del history, squared_steps, has_started, counted

        last_28_days = slice(origin_index - 27, origin_index + 1)
        prices = np.nan_to_num(data.sell_price[:, last_28_days])
        dollar_sales = self.aggregation @ (data.units_sold[:, last_28_days] * prices).sum(axis=1)
        weights = np.empty_like(dollar_sales)
        start = 0
        for size in level_sizes:
            level_sales = dollar_sales[start : start + size]
            weights[start : start + size] = level_sales / level_sales.sum()
            start += size
        self.weights = weights / len(level_sizes)

    def score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        squared_errors = (self.aggregation @ predicted - self.aggregation @ actual) ** 2
        has_scale = self.scale > 0
        rmsse = np.sqrt(squared_errors.mean(axis=1)[has_scale] / self.scale[has_scale])
        return float((self.weights[has_scale] * rmsse).sum())


def join_labels(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(first.astype(str), "_"), second.astype(str))


def run_origin(
    data: M5Data,
    origin: int,
    train_days: int,
    num_rounds: int = NUM_ROUNDS,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.Series]:
    """Train, forecast, and score one origin. origin is the M5 day number of
    the last known day, so 1913 means d_1913."""
    origin_index = origin - 1
    actual = data.units_sold[:, origin_index + 1 : origin_index + 1 + FORECAST_HORIZON]

    start_time = time.time()
    stats = fit_window_stats(data, origin_index, train_days)
    features, targets = build_training_set(data, origin_index, train_days, stats)
    log(
        f"  origin d_{origin}: {len(targets):,} training rows "
        f"({time.time() - start_time:.0f}s to build)"
    )

    dataset = lgb.Dataset(
        features,
        label=targets,
        feature_name=FEATURES,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=True,
    )
    dataset.construct()
    del features, targets
    gc.collect()

    start_time = time.time()
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=num_rounds)
    del dataset
    gc.collect()
    log(f"  trained in {time.time() - start_time:.0f}s")

    forecasts = {
        "LightGBM": recursive_forecast(data, model, origin_index, stats),
        "Seasonal naive": seasonal_naive(data, origin_index),
        "28-day average": moving_average(data, origin_index),
    }
    wrmsse = WRMSSE(data, origin_index)
    results = [
        {
            "origin": f"d_{origin}",
            "window": f"d_{origin + 1}-d_{origin + FORECAST_HORIZON}",
            "model": name,
            "RMSE": rmse(predicted, actual),
            "WAPE": wape(predicted, actual),
            "WRMSSE": wrmsse.score(predicted, actual),
        }
        for name, predicted in forecasts.items()
    ]
    importance = pd.Series(model.feature_importance(), index=FEATURES)
    return pd.DataFrame(results), importance
