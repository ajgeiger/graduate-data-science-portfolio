"""Rolling-origin backtest for the M5 sales forecast.

The original project validated the LightGBM model on rows sampled at random
from the training data, and some features (item mean and std of sales, the
price count features) were computed over the full history. Both let
information from the forecast period leak into training.

This module rebuilds the same features from src/preprocessing/Pipeline.py,
but only from data on or before each forecast origin, trains on the year
before the origin, and forecasts the next 28 days one day at a time, feeding
predictions back in as the original notebook did.

Everything is kept in numpy arrays shaped (series, day) so the run fits in a
few GB of memory.
"""

import gc
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse

HORIZON = 28
LAGS = [7, 28]
WINDOWS = [7, 28]
CAT_COLS = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]

# settings from the original Forecast notebook
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

FEATURES = (
    CAT_COLS
    + ["wday", "month", "year", "event_name_1", "event_type_1",
       "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
       "sell_price"]
    + [f"lag_{l}" for l in LAGS]
    + [f"rmean_{l}_{w}" for w in WINDOWS for l in LAGS]
    + ["week", "quarter", "mday",
       "price_nunique_items", "item_nunique_price", "item_nunique_weekly",
       "enc_item_id_mean", "enc_item_id_std"]
)
CAT_FEATURES = CAT_COLS + ["event_name_1", "event_type_1",
                           "event_name_2", "event_type_2"]


class M5Data:
    """Loads the M5 files into arrays indexed by (series, day)."""

    def __init__(self, path):
        sales = pd.read_csv(f"{path}/sales_train_evaluation.csv")
        day_cols = [c for c in sales.columns if c.startswith("d_")]
        self.n_days = len(day_cols)
        self.ids = sales["id"].values

        # integer codes for the id columns, as in DAO.getTargets
        self.codes = {}
        self.labels = {}
        for col in CAT_COLS:
            cat = sales[col].astype("category")
            self.codes[col] = cat.cat.codes.values.astype(np.int16)
            self.labels[col] = sales[col].values

        # units sold, series x day
        self.y = sales[day_cols].values.astype(np.float32)
        del sales
        gc.collect()

        cal = pd.read_csv(f"{path}/calendar.csv")
        cal = cal.iloc[: self.n_days + HORIZON].copy()
        date = pd.to_datetime(cal["date"])
        self.cal = pd.DataFrame({
            "wday": cal["wday"].astype(np.int16),
            "month": cal["month"].astype(np.int16),
            "year": cal["year"].astype(np.int16),
            "week": date.dt.isocalendar().week.astype(np.int16).values,
            "quarter": date.dt.quarter.astype(np.int16),
            "mday": date.dt.day.astype(np.int16),
            "snap_CA": cal["snap_CA"].astype(np.float32),
            "snap_TX": cal["snap_TX"].astype(np.float32),
            "snap_WI": cal["snap_WI"].astype(np.float32),
        })
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            # 0 means no event, so the codes stay non-negative for LightGBM
            self.cal[col] = (cal[col].astype("category").cat.codes + 1).astype(np.int16)
        self.wm_yr_wk = cal["wm_yr_wk"].values

        # sell price, series x day (NaN when the item was not on sale)
        prices = pd.read_csv(f"{path}/sell_prices.csv")
        weeks = np.sort(prices["wm_yr_wk"].unique())
        week_pos = {w: i for i, w in enumerate(weeks)}
        key = pd.Series(np.arange(len(self.ids)),
                        index=pd.MultiIndex.from_arrays(
                            [self.labels["store_id"], self.labels["item_id"]]))
        rows = key.reindex(pd.MultiIndex.from_arrays(
            [prices["store_id"].values, prices["item_id"].values])).values
        cols = prices["wm_yr_wk"].map(week_pos).values
        weekly = np.full((len(self.ids), len(weeks)), np.nan, dtype=np.float32)
        weekly[rows, cols] = prices["sell_price"].values.astype(np.float32)
        del prices
        day_week = np.array([week_pos[w] for w in self.wm_yr_wk])
        self.price = weekly[:, day_week]
        self.day_week = day_week
        del weekly
        gc.collect()


def day_features(data, hist, day, stats):
    """Features for every series on one day (0-based day index).

    hist holds sales up to day - 1 (actuals before the origin, predictions
    after it). stats holds the encodings fit on the training window.
    """
    n = hist.shape[0]
    out = np.empty((n, len(FEATURES)), dtype=np.float32)
    f = {name: i for i, name in enumerate(FEATURES)}
    for col in CAT_COLS:
        out[:, f[col]] = data.codes[col]
    row = data.cal.iloc[day]
    for col in ["wday", "month", "year", "event_name_1", "event_type_1",
                "event_name_2", "event_type_2", "snap_CA", "snap_TX",
                "snap_WI", "week", "quarter", "mday"]:
        out[:, f[col]] = row[col]
    price = data.price[:, day]
    out[:, f["sell_price"]] = price

    # lag_k is sales k days back; rmean_k_w is the mean of the w days ending
    # at that lag, which matches the pandas shift and rolling in Pipeline.py
    for l in LAGS:
        out[:, f[f"lag_{l}"]] = hist[:, day - l]
        for w in WINDOWS:
            out[:, f[f"rmean_{l}_{w}"]] = hist[:, day - l - w + 1: day - l + 1].mean(axis=1)

    out[:, f["price_nunique_items"]] = stats["price_nunique_items"]
    out[:, f["item_nunique_price"]] = stats["item_nunique_price"](price)
    out[:, f["item_nunique_weekly"]] = stats["item_nunique_weekly"][:, int(row["week"])]
    out[:, f["enc_item_id_mean"]] = stats["enc_mean"]
    out[:, f["enc_item_id_std"]] = stats["enc_std"]
    return out


def fit_stats(data, origin, train_days):
    """Encodings from Pipeline.buildGeneralFeatures, using only the training
    window (days origin - train_days + 1 .. origin, 0-based, inclusive)."""
    start = origin - train_days + 1
    y = data.y[:, start: origin + 1]
    price = data.price[:, start: origin + 1]
    on_sale = ~np.isnan(price)
    store = data.codes["store_id"]
    item = data.codes["item_id"]

    # number of distinct prices each store/item had
    pn = np.array([len(np.unique(p[~np.isnan(p)])) for p in price], dtype=np.float32)

    # number of items each store sold at a given price
    s_idx, d_idx = np.nonzero(on_sale)
    pairs = pd.DataFrame({"store": store[s_idx], "price": price[s_idx, d_idx],
                          "item": item[s_idx]})
    inp = pairs.groupby(["store", "price"])["item"].nunique()
    del pairs

    def item_nunique_price(day_price):
        keys = pd.MultiIndex.from_arrays([store, day_price])
        return inp.reindex(keys).values.astype(np.float32)

    # number of items each store had on sale in a given week of the year
    weeks = data.cal["week"].values[start: origin + 1]
    frame = pd.DataFrame({"store": store[s_idx], "week": weeks[d_idx],
                          "item": item[s_idx]})
    counts = frame.groupby(["store", "week"])["item"].nunique()
    del frame
    by_store = np.full((store.max() + 1, 54), np.nan, dtype=np.float32)
    for (s, w), v in counts.items():
        by_store[s, w] = v
    inw = by_store[store]

    # item-level mean and std of sales (across stores), on-sale days only
    ys = np.where(on_sale, y, np.nan)
    frame = pd.DataFrame({"item": item,
                          "sum": np.nansum(ys, axis=1),
                          "sq": np.nansum(ys ** 2, axis=1),
                          "n": on_sale.sum(axis=1)})
    g = frame.groupby("item")[["sum", "sq", "n"]].sum()
    mean = g["sum"] / g["n"]
    var = (g["sq"] - g["n"] * mean ** 2) / (g["n"] - 1)
    enc_mean = mean.reindex(item).values.astype(np.float32)
    enc_std = np.sqrt(var.clip(lower=0)).reindex(item).values.astype(np.float32)

    return {"price_nunique_items": pn, "item_nunique_price": item_nunique_price,
            "item_nunique_weekly": inw, "enc_mean": enc_mean,
            "enc_std": enc_std}


def train_matrix(data, origin, train_days, stats):
    """Stack day_features for every on-sale series/day in the window."""
    start = origin - train_days + 1
    on_sale = ~np.isnan(data.price[:, start: origin + 1])
    total = int(on_sale.sum())
    X = np.empty((total, len(FEATURES)), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    hist = data.y
    pos = 0
    for k, day in enumerate(range(start, origin + 1)):
        mask = on_sale[:, k]
        block = day_features(data, hist, day, stats)[mask]
        X[pos: pos + len(block)] = block
        y[pos: pos + len(block)] = data.y[mask, day]
        pos += len(block)
    return X, y


def forecast(data, model, origin, stats):
    """Recursive 28-day forecast after origin (0-based last known day)."""
    hist = np.concatenate(
        [data.y[:, : origin + 1],
         np.zeros((data.y.shape[0], HORIZON), dtype=np.float32)], axis=1)
    for h in range(HORIZON):
        day = origin + 1 + h
        X = day_features(data, hist, day, stats)
        hist[:, day] = np.clip(model.predict(X), 0, None)
    return hist[:, origin + 1:]


def seasonal_naive(data, origin):
    """Repeat the last observed week."""
    last_week = data.y[:, origin - 6: origin + 1]
    return np.tile(last_week, (1, HORIZON // 7))


def moving_average(data, origin, window=28):
    """Flat forecast at the mean of the last 28 days."""
    m = data.y[:, origin - window + 1: origin + 1].mean(axis=1, keepdims=True)
    return np.repeat(m, HORIZON, axis=1)


def rmse(pred, actual):
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def wape(pred, actual):
    return float(np.abs(pred - actual).sum() / actual.sum())


class WRMSSE:
    """M5 accuracy metric over the 12 aggregation levels (42,840 series),
    with weights from dollar sales in the 28 days up to the origin and scales
    from in-sample one-day naive errors, as in the competition guide."""

    def __init__(self, data, origin):
        n = data.y.shape[0]
        lab = data.labels
        groups = [
            np.zeros(n, dtype=object),
            lab["state_id"], lab["store_id"], lab["cat_id"], lab["dept_id"],
            _join(lab["state_id"], lab["cat_id"]),
            _join(lab["state_id"], lab["dept_id"]),
            _join(lab["store_id"], lab["cat_id"]),
            _join(lab["store_id"], lab["dept_id"]),
            lab["item_id"],
            _join(lab["item_id"], lab["state_id"]),
            data.ids,
        ]
        mats = []
        for g in groups:
            codes, _ = pd.factorize(g)
            mats.append(sparse.csr_matrix(
                (np.ones(n, dtype=np.float32), (codes, np.arange(n))),
                shape=(codes.max() + 1, n)))
        self.S = sparse.vstack(mats).tocsr()
        level_sizes = [m.shape[0] for m in mats]

        hist = self.S @ data.y[:, : origin + 1]
        started = np.cumsum(hist, axis=1) > 0
        diff2 = np.diff(hist, axis=1) ** 2
        valid = started[:, :-1]
        self.scale = (diff2 * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        del hist, diff2, started, valid

        price = np.nan_to_num(data.price[:, origin - 27: origin + 1])
        dollars = self.S @ (data.y[:, origin - 27: origin + 1] * price).sum(axis=1)
        w = np.empty_like(dollars)
        pos = 0
        for size in level_sizes:
            seg = dollars[pos: pos + size]
            w[pos: pos + size] = seg / seg.sum()
            pos += size
        self.weight = w / len(level_sizes)

    def score(self, pred, actual):
        err = (self.S @ pred - self.S @ actual) ** 2
        ok = self.scale > 0
        rmsse = np.sqrt(err.mean(axis=1)[ok] / self.scale[ok])
        return float((self.weight[ok] * rmsse).sum())


def _join(a, b):
    return np.char.add(np.char.add(a.astype(str), "_"), b.astype(str))


def run_origin(data, origin, train_days, num_rounds=NUM_ROUNDS, log=print):
    """Train, forecast, and score one origin. origin is the 1-based M5 day
    number of the last known day (for example 1913 means d_1913)."""
    o = origin - 1
    actual = data.y[:, o + 1: o + 1 + HORIZON]
    t0 = time.time()
    stats = fit_stats(data, o, train_days)
    X, y = train_matrix(data, o, train_days, stats)
    log(f"  origin d_{origin}: {len(y):,} training rows "
        f"({time.time() - t0:.0f}s to build)")
    ds = lgb.Dataset(X, label=y, feature_name=FEATURES,
                     categorical_feature=CAT_FEATURES, free_raw_data=True)
    ds.construct()
    del X, y
    gc.collect()
    t0 = time.time()
    model = lgb.train(LGB_PARAMS, ds, num_boost_round=num_rounds)
    del ds
    gc.collect()
    log(f"  trained in {time.time() - t0:.0f}s")

    preds = {
        "LightGBM": forecast(data, model, o, stats),
        "Seasonal naive": seasonal_naive(data, o),
        "28-day average": moving_average(data, o),
    }
    metric = WRMSSE(data, o)
    rows = []
    for name, p in preds.items():
        rows.append({"origin": f"d_{origin}",
                     "window": f"d_{origin + 1}-d_{origin + HORIZON}",
                     "model": name,
                     "RMSE": rmse(p, actual),
                     "WAPE": wape(p, actual),
                     "WRMSSE": metric.score(p, actual)})
    importance = pd.Series(model.feature_importance(), index=FEATURES)
    return pd.DataFrame(rows), importance
