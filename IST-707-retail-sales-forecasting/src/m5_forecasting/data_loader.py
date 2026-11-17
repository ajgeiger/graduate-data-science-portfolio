"""Loads the M5 files and joins sales, calendar, and prices into one table.

Parts of the loading code follow these public Kaggle notebooks:
kernelgenerator/m5-accuracy-tweedie-is-back, kyakovlev/m5-custom-features,
kyakovlev/m5-lags-features, and kyakovlev/m5-simple-fe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]
FORECAST_HORIZON = 28


class M5DataLoader:
    """Reads the raw M5 csv files from a folder."""

    def __init__(
        self,
        path: str,
        first_day: int = 1000,
        last_train_day: int = 1913,
        max_lag_days: int = 57,
        verbose: bool = True,
    ):
        self.path = path
        self.first_day = first_day
        self.last_train_day = last_train_day
        self.max_lag_days = max_lag_days
        self.verbose = verbose

    def load(self, is_train: bool = True, nrows: int | None = None) -> pd.DataFrame:
        """Return one row per item, store, and day with calendar and price columns."""
        prices = self.load_sell_prices()
        calendar = self.load_calendar()
        sales = self.load_sales(is_train, nrows)

        day_columns = [column for column in sales.columns if column.startswith("d_")]
        sales = pd.melt(
            sales,
            id_vars=ID_COLUMNS,
            value_vars=day_columns,
            var_name="d",
            value_name="sales",
        )
        sales = sales.merge(calendar, on="d")
        sales = sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"])
        return sales

    def load_calendar(self) -> pd.DataFrame:
        dtypes = {
            "event_name_1": "category",
            "event_name_2": "category",
            "event_type_1": "category",
            "event_type_2": "category",
            "weekday": "category",
            "wm_yr_wk": "int16",
            "wday": "int16",
            "month": "int16",
            "year": "int16",
            "snap_CA": "float32",
            "snap_TX": "float32",
            "snap_WI": "float32",
        }
        calendar = pd.read_csv(f"{self.path}/calendar.csv", dtype=dtypes)
        return encode_categories(calendar)

    def load_sell_prices(self) -> pd.DataFrame:
        dtypes = {
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int16",
            "sell_price": "float32",
        }
        prices = pd.read_csv(f"{self.path}/sell_prices.csv", dtype=dtypes)
        return encode_categories(prices)

    def load_sales(self, is_train: bool, nrows: int | None) -> pd.DataFrame:
        """Daily sales in wide format. For test data, adds empty columns for
        the 28 days being forecast."""
        day_columns = self.sales_day_columns(is_train)
        dtypes = {column: "float32" for column in day_columns}
        dtypes.update({column: "category" for column in ID_COLUMNS if column != "id"})

        sales = pd.read_csv(
            f"{self.path}/sales_train_validation.csv",
            nrows=nrows,
            usecols=ID_COLUMNS + day_columns,
            dtype=dtypes,
        )
        for column in ID_COLUMNS:
            if column != "id":
                sales[column] = sales[column].cat.codes.astype("int16")
                sales[column] -= sales[column].min()

        if not is_train:
            for day in range(self.last_train_day + 1, self.last_train_day + FORECAST_HORIZON + 1):
                sales[f"d_{day}"] = np.nan
        return sales

    def sales_day_columns(self, is_train: bool) -> list[str]:
        """Training uses every day from first_day on. Test only needs enough
        history to build the lag features."""
        start_day = 1 if is_train else self.last_train_day - self.max_lag_days
        start_day = max(start_day, self.first_day)
        return [f"d_{day}" for day in range(start_day, self.last_train_day + 1)]


def encode_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace category columns with small integer codes to save memory."""
    for column, dtype in frame.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            frame[column] = frame[column].cat.codes.astype("int16")
            frame[column] -= frame[column].min()
    return frame
