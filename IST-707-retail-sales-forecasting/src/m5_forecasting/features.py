"""Feature pipeline used by the original forecast, clustering, and
classification notebooks.

The lag, rolling mean, and price features follow the public Kaggle notebooks
listed in data_loader.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from m5_forecasting.data_loader import FORECAST_HORIZON, M5DataLoader

LAGS = [7, 28]
WINDOWS = [7, 28]

# pandas datetime attribute used to fill each date feature
DATE_FEATURES = {
    "wday": "weekday",
    "week": "week",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
    "mday": "day",
}


class FeaturePipeline(M5DataLoader):
    """Builds the training, test, and classification feature sets."""

    def __init__(
        self,
        path: str,
        first_day: int = 1000,
        last_train_day: int = 1913,
        verbose: bool = True,
    ):
        super().__init__(path, first_day, last_train_day, max_lag_days=57, verbose=verbose)
        self.forecast_start_date = datetime(2016, 4, 25)

    def build_feature_set(self) -> pd.DataFrame:
        return add_features(self.load(is_train=True))

    def build_test_data(self) -> pd.DataFrame:
        test = self.load(is_train=False)
        test["date"] = pd.to_datetime(test["date"])
        return test

    def build_test_features(
        self, test: pd.DataFrame, day_offset: int
    ) -> tuple[pd.DataFrame, datetime]:
        """Features for one forecast day, using the history in test (which
        includes earlier predictions)."""
        day = self.forecast_start_date + timedelta(days=day_offset)
        history_start = day - timedelta(days=self.max_lag_days)
        window = test[(test["date"] >= history_start) & (test["date"] <= day)].copy()
        window = add_features(window)
        return window.loc[window["date"] == day].copy(), day

    def build_classification_feature_set(self) -> pd.DataFrame:
        """One row per store, product category, and week, labeled with the
        weekday that had the most sales."""
        features = self.build_feature_set()
        week_group = ["store_id", "id", "week", "year"]

        features["id"] = features["id"].apply(lambda item_id: item_id.split("_")[0])

        is_best_day = features.groupby(week_group)["sales"].transform("max") == features["sales"]
        features["best_day"] = is_best_day * (features["wday"] + 1)
        features["best_day"] = features.groupby(week_group)["best_day"].transform("max")

        first_day_of_week = features.groupby(week_group)["date"].transform("min")
        return features.loc[first_day_of_week == features["date"]].copy()

    def create_submission(self, test: pd.DataFrame) -> pd.DataFrame:
        """Reshape forecasts to the Kaggle format: one row per id, columns F1 to F28."""
        forecast = test.loc[test["date"] >= self.forecast_start_date, ["id", "sales"]].copy()
        forecast["F"] = [f"F{rank}" for rank in forecast.groupby("id")["id"].cumcount() + 1]
        submission = forecast.set_index(["id", "F"]).unstack()["sales"][forecast_columns()]
        submission = submission.reset_index()
        submission.fillna(0.0, inplace=True)
        submission.sort_values("id", inplace=True)
        submission.reset_index(drop=True, inplace=True)
        return submission


def forecast_columns() -> list[str]:
    return [f"F{day}" for day in range(1, FORECAST_HORIZON + 1)]


def add_features(sales: pd.DataFrame) -> pd.DataFrame:
    add_lag_features(sales)
    add_rolling_mean_features(sales)
    add_date_features(sales)
    add_price_and_item_features(sales)
    return sales


def add_lag_features(sales: pd.DataFrame) -> None:
    for lag in LAGS:
        sales[f"lag_{lag}"] = sales[["id", "sales"]].groupby("id")["sales"].shift(lag)


def add_rolling_mean_features(sales: pd.DataFrame) -> None:
    """rmean_{lag}_{window} is the mean of lag_{lag} over the last window days."""
    for window in WINDOWS:
        for lag in LAGS:
            lag_column = f"lag_{lag}"
            sales[f"rmean_{lag}_{window}"] = (
                sales[["id", lag_column]]
                .groupby("id")[lag_column]
                .transform(lambda series, window=window: series.rolling(window).mean())
            )


def add_date_features(sales: pd.DataFrame) -> None:
    sales["date"] = pd.to_datetime(sales["date"])
    for feature, attribute in DATE_FEATURES.items():
        if feature in sales.columns:
            sales[feature] = sales[feature].astype("int16")
        elif attribute == "week":
            sales[feature] = sales["date"].dt.isocalendar().week.astype("int16")
        else:
            sales[feature] = getattr(sales["date"].dt, attribute).astype("int16")


def add_price_and_item_features(sales: pd.DataFrame) -> None:
    """Price history and product line features, plus item level sales mean and std."""
    store_item = sales.groupby(["store_id", "item_id"])
    sales["price_nunique_items"] = store_item["sell_price"].transform("nunique")

    store_price = sales.groupby(["store_id", "sell_price"])
    sales["item_nunique_price"] = store_price["item_id"].transform("nunique")

    store_week = sales.groupby(["store_id", "week"])
    sales["item_nunique_weekly"] = store_week["item_id"].transform("nunique")

    item_sales = sales.groupby(["item_id"])["sales"]
    sales["enc_item_id_mean"] = item_sales.transform("mean").astype(np.float16)
    sales["enc_item_id_std"] = item_sales.transform("std").astype(np.float16)
