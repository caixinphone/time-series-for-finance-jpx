import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import lightgbm as lgb
except Exception:
    lgb = None

INPUT_ROOT = Path("../input")
RANDOM_STATE = 42
MAX_TRAIN_ROWS = 900_000
HISTORY_TAIL_DAYS = 45
ROLLING_WEIGHT = 1.5
CLOSE_DIFF_FEATURE_COLUMNS = ["close_diff1"]
CLOSE_DIFF_HISTORY_DAYS = 100


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


SIMPLE_FEATURE_COLUMNS = [
    "SecuritiesCode",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "AdjustmentFactor",
    "ExpectedDividend",
    "SupervisionFlag",
    "dividend_flag",
    "dayofweek",
    "month",
    "high_low_spread",
    "close_open_return",
    "33SectorCode",
    "17SectorCode",
    "NewIndexSeriesSizeCode",
    "market_cap_log",
    "issued_shares_log",
    "Universe0",
]


def make_simple_features(prices, stock_list):
    df = prices.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["SecuritiesCode"] = df["SecuritiesCode"].astype("int32")
    df = df.merge(stock_list, on="SecuritiesCode", how="left")
    for col in ["Open", "High", "Low", "Close", "Volume", "AdjustmentFactor", "ExpectedDividend"]:
        if col not in df.columns:
            df[col] = np.nan
    if "SupervisionFlag" not in df.columns:
        df["SupervisionFlag"] = False
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0.0)
    df["SupervisionFlag"] = df["SupervisionFlag"].fillna(False).astype("int8")
    df["AdjustmentFactor"] = df["AdjustmentFactor"].fillna(1.0)
    df["dividend_flag"] = (df["ExpectedDividend"] != 0).astype("int8")
    df["dayofweek"] = df["Date"].dt.dayofweek.astype("int8")
    df["month"] = df["Date"].dt.month.astype("int8")
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_open_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["market_cap_log"] = np.log1p(df["MarketCapitalization"])
    df["issued_shares_log"] = np.log1p(df["IssuedShares"])
    df[SIMPLE_FEATURE_COLUMNS] = df[SIMPLE_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return df


def make_model(n_estimators=260, learning_rate=0.05, num_leaves=63, random_state=RANDOM_STATE):
    if lgb is not None:
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=n_estimators,
        max_leaf_nodes=63,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
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


def make_close_diff_features(prices):
    df = add_adjusted_close(prices)
    df = df.sort_values(["SecuritiesCode", "Date"]).copy()
    df["close_diff1"] = df.groupby("SecuritiesCode", sort=False)["AdjustedClose"].diff(1)
    df[CLOSE_DIFF_FEATURE_COLUMNS] = df[CLOSE_DIFF_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


DATA_ROOT = find_data_root()
print("Using data root:", DATA_ROOT)
prepare_competition_api(DATA_ROOT)

stock_list = read_stock_list(DATA_ROOT / "stock_list.csv")
all_prices = load_all_prices(DATA_ROOT)
labeled_prices = all_prices.dropna(subset=["Target"]).copy()

if lgb is not None:
    training_prices = labeled_prices.loc[labeled_prices["Date"] <= "2019-12-31"].copy()
    close_diff_training_features = make_close_diff_features(training_prices).dropna(subset=["Target"]).copy()
    close_diff_model = lgb.LGBMRegressor(seed=RANDOM_STATE, n_jobs=-1, verbose=-1)
    close_diff_model.fit(
        close_diff_training_features[CLOSE_DIFF_FEATURE_COLUMNS],
        close_diff_training_features["Target"],
    )
    model = None
    simple_model = None
else:
    training_prices = labeled_prices.loc[labeled_prices["Date"] >= "2019-01-01"].copy()
    training_features = make_features(training_prices, stock_list).dropna(subset=["Target"]).copy()
    if len(training_features) > MAX_TRAIN_ROWS:
        training_features = training_features.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE).sort_values(
            ["Date", "SecuritiesCode"]
        )
    model = make_model()
    model.fit(training_features[FEATURE_COLUMNS], training_features["Target"])

    simple_training_features = make_simple_features(training_prices, stock_list).dropna(subset=["Target"]).copy()
    if len(simple_training_features) > MAX_TRAIN_ROWS:
        simple_training_features = simple_training_features.sample(
            MAX_TRAIN_ROWS, random_state=RANDOM_STATE
        ).sort_values(["Date", "SecuritiesCode"])
    simple_model = make_model()
    simple_model.fit(simple_training_features[SIMPLE_FEATURE_COLUMNS], simple_training_features["Target"])

if lgb is not None:
    close_diff_history = tail_by_code(
        all_prices[["Date", "SecuritiesCode", "Close", "AdjustmentFactor"]].drop_duplicates(["Date", "SecuritiesCode"]),
        n=CLOSE_DIFF_HISTORY_DAYS,
    )
    first_close_diff_batch = True
else:
    history_tail = tail_by_code(all_prices.drop(columns=["Target"]))

import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for prices, options, financials, trades, secondary_prices, sample_prediction in iter_test:
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prediction_dates = set(prices["Date"].unique())
    if lgb is not None:
        price_tail = prices[["Date", "SecuritiesCode", "Close", "AdjustmentFactor"]].copy()
        if first_close_diff_batch:
            close_diff_history = close_diff_history.loc[close_diff_history["Date"] < prices["Date"].min()].copy()
            first_close_diff_batch = False
        close_diff_history = pd.concat([close_diff_history, price_tail], ignore_index=True, sort=False)
        test_features = make_close_diff_features(close_diff_history)
        test_features = test_features.loc[test_features["Date"].isin(prediction_dates)].copy()
        predictions = close_diff_model.predict(test_features[CLOSE_DIFF_FEATURE_COLUMNS])
        prediction_frame = rank_predictions(sample_prediction, predictions)
        env.predict(prediction_frame)
        close_diff_history = tail_by_code(close_diff_history, n=CLOSE_DIFF_HISTORY_DAYS)
    else:
        feature_source = pd.concat([history_tail, prices], ignore_index=True, sort=False)
        test_features = make_features(feature_source, stock_list)
        test_features = test_features.loc[test_features["Date"].isin(prediction_dates)].copy()
        rolling_predictions = model.predict(test_features[FEATURE_COLUMNS])
        simple_features = make_simple_features(prices, stock_list)
        simple_predictions = simple_model.predict(simple_features[SIMPLE_FEATURE_COLUMNS])
        predictions = ROLLING_WEIGHT * rolling_predictions + (1.0 - ROLLING_WEIGHT) * simple_predictions
        prediction_frame = rank_predictions(sample_prediction, predictions)
        env.predict(prediction_frame)
        history_tail = tail_by_code(pd.concat([history_tail, prices], ignore_index=True, sort=False))
