import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


INPUT_ROOT = Path("../input")
RANDOM_STATE = 42
MAX_TRAIN_ROWS = 900_000
HISTORY_TAIL_DAYS = 45


def read_stock_list(path):
    usecols = [
        "SecuritiesCode",
        "33SectorCode",
        "17SectorCode",
        "NewIndexSeriesSizeCode",
        "MarketCapitalization",
        "IssuedShares",
        "Universe0",
    ]
    stock_list = pd.read_csv(path, usecols=usecols)
    stock_list["SecuritiesCode"] = stock_list["SecuritiesCode"].astype("int32")
    for col in ["33SectorCode", "17SectorCode", "NewIndexSeriesSizeCode"]:
        stock_list[col] = pd.to_numeric(stock_list[col], errors="coerce")
    stock_list["Universe0"] = stock_list["Universe0"].fillna(False).astype("int8")
    return stock_list


def find_data_root():
    candidates = [
        INPUT_ROOT / "jpx-tokyo-stock-exchange-prediction",
        INPUT_ROOT / "jpx-tokyo-market-prediction",
    ]
    for candidate in candidates:
        if (candidate / "stock_list.csv").exists():
            return candidate
    matches = list(INPUT_ROOT.glob("**/stock_list.csv"))
    if matches:
        return matches[0].parent
    visible = [str(p) for p in INPUT_ROOT.glob("*")]
    raise FileNotFoundError("Could not find stock_list.csv. Visible inputs: %s" % visible)


def prepare_competition_api(data_root):
    sys.path.insert(0, str(data_root))


def read_prices(path, include_target=True):
    columns = [
        "RowId",
        "Date",
        "SecuritiesCode",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "AdjustmentFactor",
        "ExpectedDividend",
        "SupervisionFlag",
        "Target",
    ]
    if not include_target:
        columns = [c for c in columns if c != "Target"]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in columns if c in available]
    prices = pd.read_csv(path, usecols=usecols, parse_dates=["Date"])
    prices["SecuritiesCode"] = prices["SecuritiesCode"].astype("int32")
    return prices


def load_all_prices(data_root):
    train = read_prices(data_root / "train_files" / "stock_prices.csv", include_target=True)
    parts = [train]
    supplemental_path = data_root / "supplemental_files" / "stock_prices.csv"
    if supplemental_path.exists():
        supplemental = read_prices(supplemental_path, include_target=True)
        parts.append(supplemental)
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(["Date", "SecuritiesCode"])


def add_adjusted_close(prices):
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    factor = prices["AdjustmentFactor"].fillna(1.0).replace(0, 1.0)
    prices["_reverse_factor"] = factor
    prices["_cum_adjustment"] = (
        prices.sort_values(["SecuritiesCode", "Date"], ascending=[True, False])
        .groupby("SecuritiesCode", sort=False)["_reverse_factor"]
        .cumprod()
    )
    prices["AdjustedClose"] = prices["Close"] * prices["_cum_adjustment"]
    return prices.drop(columns=["_reverse_factor", "_cum_adjustment"]).sort_values(["SecuritiesCode", "Date"])


def add_price_features(prices):
    prices = add_adjusted_close(prices)
    group = prices.groupby("SecuritiesCode", sort=False)
    prices["ret_1"] = group["AdjustedClose"].pct_change(1)
    for window in [2, 5, 10, 20]:
        prices["ret_%d" % window] = group["AdjustedClose"].pct_change(window)

    shifted_ret = group["ret_1"].shift(1)
    for window in [5, 10, 20]:
        min_periods = max(2, window // 2)
        prices["ret_mean_%d" % window] = (
            shifted_ret.groupby(prices["SecuritiesCode"], sort=False)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        prices["ret_std_%d" % window] = (
            shifted_ret.groupby(prices["SecuritiesCode"], sort=False)
            .rolling(window, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
        )

    shifted_volume = group["Volume"].shift(1)
    prices["volume_mean_20"] = (
        shifted_volume.groupby(prices["SecuritiesCode"], sort=False)
        .rolling(20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    prices["volume_ratio_20"] = prices["Volume"] / prices["volume_mean_20"]
    prices["high_low_spread"] = (prices["High"] - prices["Low"]) / prices["Close"]
    prices["close_open_return"] = (prices["Close"] - prices["Open"]) / prices["Open"]
    prices["dividend_flag"] = prices["ExpectedDividend"].notna().astype("int8")
    prices["ExpectedDividend"] = prices["ExpectedDividend"].fillna(0.0)
    prices["SupervisionFlag"] = prices["SupervisionFlag"].fillna(False).astype("int8")
    prices["dayofweek"] = prices["Date"].dt.dayofweek.astype("int8")
    prices["month"] = prices["Date"].dt.month.astype("int8")
    return prices


FEATURE_COLUMNS = [
    "SecuritiesCode",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "AdjustedClose",
    "ExpectedDividend",
    "SupervisionFlag",
    "ret_1",
    "ret_2",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_mean_5",
    "ret_mean_10",
    "ret_mean_20",
    "ret_std_5",
    "ret_std_10",
    "ret_std_20",
    "volume_ratio_20",
    "high_low_spread",
    "close_open_return",
    "dividend_flag",
    "dayofweek",
    "month",
    "33SectorCode",
    "17SectorCode",
    "NewIndexSeriesSizeCode",
    "market_cap_log",
    "issued_shares_log",
    "Universe0",
]


def make_features(prices, stock_list):
    features = add_price_features(prices)
    features = features.merge(stock_list, on="SecuritiesCode", how="left")
    features["market_cap_log"] = np.log1p(features["MarketCapitalization"])
    features["issued_shares_log"] = np.log1p(features["IssuedShares"])
    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return features


def make_model():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=260,
        max_leaf_nodes=63,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


def rank_predictions(sample_prediction, predictions):
    ranked = sample_prediction[["Date", "SecuritiesCode"]].copy()
    ranked["prediction"] = predictions
    ranked = ranked.sort_values(["Date", "prediction", "SecuritiesCode"], ascending=[True, False, True])
    ranked["Rank"] = ranked.groupby("Date").cumcount().astype("int32")
    out = sample_prediction.merge(ranked[["Date", "SecuritiesCode", "Rank"]], on=["Date", "SecuritiesCode"], how="left")
    if "Rank_x" in out.columns:
        out = out.drop(columns=["Rank_x"]).rename(columns={"Rank_y": "Rank"})
    out["Rank"] = out["Rank"].fillna(0).astype("int32")
    return out


def tail_by_code(prices, n=HISTORY_TAIL_DAYS):
    return prices.sort_values(["SecuritiesCode", "Date"]).groupby("SecuritiesCode", sort=False).tail(n)


DATA_ROOT = find_data_root()
print("Using data root:", DATA_ROOT)
prepare_competition_api(DATA_ROOT)

stock_list = read_stock_list(DATA_ROOT / "stock_list.csv")
all_prices = load_all_prices(DATA_ROOT)
training_prices = all_prices.dropna(subset=["Target"]).copy()
training_prices = training_prices.loc[training_prices["Date"] >= "2019-01-01"].copy()
training_features = make_features(training_prices, stock_list).dropna(subset=["Target"]).copy()
if len(training_features) > MAX_TRAIN_ROWS:
    training_features = training_features.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE).sort_values(
        ["Date", "SecuritiesCode"]
    )

model = make_model()
model.fit(training_features[FEATURE_COLUMNS], training_features["Target"])

history_tail = tail_by_code(all_prices.drop(columns=["Target"]))

import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for prices, options, financials, trades, secondary_prices, sample_prediction in iter_test:
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prediction_dates = set(prices["Date"].unique())
    feature_source = pd.concat([history_tail, prices], ignore_index=True, sort=False)
    test_features = make_features(feature_source, stock_list)
    test_features = test_features.loc[test_features["Date"].isin(prediction_dates)].copy()
    predictions = model.predict(test_features[FEATURE_COLUMNS])
    prediction_frame = rank_predictions(sample_prediction, predictions)
    env.predict(prediction_frame)
    history_tail = tail_by_code(pd.concat([history_tail, prices], ignore_index=True, sort=False))
