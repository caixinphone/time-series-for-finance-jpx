import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

INPUT_ROOT = Path("../input")
LGB_FEATURES = [
    "High",
    "Low",
    "Open",
    "Close",
    "Volume",
    "return_1month",
    "return_2month",
    "return_3month",
    "volatility_1month",
    "volatility_2month",
    "volatility_3month",
    "MA_gap_1month",
    "MA_gap_2month",
    "MA_gap_3month",
]
PRICE_COLUMNS = ["Date", "SecuritiesCode", "Close", "AdjustmentFactor", "ExpectedDividend"]
LGB_RANK_WEIGHT = 0.65
RULE_RANK_WEIGHT = 0.35


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


def add_lgb_features(feats):
    feats = feats.copy()
    feats["return_1month"] = feats["Close"].pct_change(20)
    feats["return_2month"] = feats["Close"].pct_change(40)
    feats["return_3month"] = feats["Close"].pct_change(60)
    feats["volatility_1month"] = np.log(feats["Close"]).diff().rolling(20).std()
    feats["volatility_2month"] = np.log(feats["Close"]).diff().rolling(40).std()
    feats["volatility_3month"] = np.log(feats["Close"]).diff().rolling(60).std()
    feats["MA_gap_1month"] = feats["Close"] / feats["Close"].rolling(20).mean()
    feats["MA_gap_2month"] = feats["Close"] / feats["Close"].rolling(40).mean()
    feats["MA_gap_3month"] = feats["Close"] / feats["Close"].rolling(60).mean()
    return feats


def fill_nan_inf(df):
    return df.fillna(0).replace([np.inf, -np.inf], 0)


def feval_pearsonr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return "pearsonr", stats.pearsonr(y_true, y_pred)[0], True


def rank_desc(df, score_col):
    out = df.copy()
    out["Rank"] = out.groupby("Date")[score_col].rank(ascending=False, method="first") - 1
    out["Rank"] = out["Rank"].astype("int32")
    return out


def adjust_price(price):
    price = price.copy()
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        df = df.sort_values("Date", ascending=False)
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        df.loc[:, "AdjustedClose"] = (df["CumulativeAdjustmentFactor"] * df["Close"]).map(
            lambda x: float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        )
        df = df.sort_values("Date")
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    price.set_index("Date", inplace=True)
    return price


def get_rule_features(price, code):
    close_col = "AdjustedClose"
    feats = price.loc[price["SecuritiesCode"] == code, ["SecuritiesCode", close_col, "ExpectedDividend"]].copy()
    feats["return_1day"] = feats[close_col].pct_change(1)
    feats["ExpectedDividend"] = feats["ExpectedDividend"].mask(feats["ExpectedDividend"] > 0, 1)
    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    feats = feats.drop([close_col], axis=1)
    return feats


DATA_ROOT = find_data_root()
sys.path.insert(0, str(DATA_ROOT))

train = pd.read_csv(DATA_ROOT / "train_files" / "stock_prices.csv", parse_dates=["Date"])
train = train.drop(columns=["RowId", "ExpectedDividend", "AdjustmentFactor", "SupervisionFlag"]).dropna().reset_index(
    drop=True
)
train = fill_nan_inf(add_lgb_features(train))

target_spread = train.groupby("SecuritiesCode")["Target"].max() - train.groupby("SecuritiesCode")["Target"].min()
list_spred_h = list(target_spread.sort_values()[:1000].index)
list_spred_l = list(target_spread.sort_values()[1000:].index)

params_lgb = {
    "learning_rate": 0.005,
    "metric": "None",
    "objective": "regression",
    "boosting": "gbdt",
    "verbosity": 0,
    "n_jobs": -1,
    "force_col_wise": True,
}

tr_dataset = lgb.Dataset(
    train[train["SecuritiesCode"].isin(list_spred_h)][LGB_FEATURES],
    train[train["SecuritiesCode"].isin(list_spred_h)]["Target"],
    feature_name=LGB_FEATURES,
)
vl_dataset = lgb.Dataset(
    train[train["SecuritiesCode"].isin(list_spred_l)][LGB_FEATURES],
    train[train["SecuritiesCode"].isin(list_spred_l)]["Target"],
    feature_name=LGB_FEATURES,
)
model = lgb.train(
    params=params_lgb,
    train_set=tr_dataset,
    valid_sets=[tr_dataset, vl_dataset],
    num_boost_round=3000,
    feval=feval_pearsonr,
    callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False), lgb.log_evaluation(period=0)],
)

df_price_raw = pd.read_csv(DATA_ROOT / "train_files" / "stock_prices.csv")
df_price_raw = df_price_raw[PRICE_COLUMNS]
df_price_supplemental = pd.read_csv(DATA_ROOT / "supplemental_files" / "stock_prices.csv")
df_price_supplemental = df_price_supplemental[PRICE_COLUMNS]
df_price_raw = pd.concat([df_price_raw, df_price_supplemental])
df_price_raw = df_price_raw.loc[df_price_raw["Date"] >= "2022-07-01"]

import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

counter = 0
for prices, options, financials, trades, secondary_prices, sample_prediction in iter_test:
    current_date = prices["Date"].iloc[0]

    lgb_prices = fill_nan_inf(add_lgb_features(prices))
    lgb_prices["Target"] = model.predict(lgb_prices[LGB_FEATURES])
    lgb_prices["target_mean"] = lgb_prices.groupby("Date")["Target"].transform("median")
    lgb_prices.loc[lgb_prices["SecuritiesCode"].isin(list_spred_h), "Target"] = lgb_prices["target_mean"]
    lgb_rank = rank_desc(lgb_prices[["Date", "SecuritiesCode", "Target"]], "Target")

    if counter == 0:
        df_price_raw = df_price_raw.loc[df_price_raw["Date"] < current_date]
    df_price_raw = pd.concat([df_price_raw, prices[PRICE_COLUMNS]])
    df_price = adjust_price(df_price_raw)
    codes = sorted(prices["SecuritiesCode"].unique())
    rule_feature = pd.concat([get_rule_features(df_price, code) for code in codes])
    rule_feature = rule_feature.loc[rule_feature.index == current_date]
    rule_feature.loc[:, "rule_score"] = rule_feature["return_1day"] + rule_feature["ExpectedDividend"] * 100
    rule_feature = rule_feature.sort_values("rule_score", ascending=True).drop_duplicates(subset=["SecuritiesCode"])
    rule_feature.loc[:, "rule_rank"] = np.arange(len(rule_feature))

    combined = sample_prediction[["Date", "SecuritiesCode"]].copy()
    combined = combined.merge(lgb_rank[["Date", "SecuritiesCode", "Rank"]], on=["Date", "SecuritiesCode"], how="left")
    combined = combined.merge(
        rule_feature[["SecuritiesCode", "rule_rank"]],
        on="SecuritiesCode",
        how="left",
    )
    combined["combined_rank_score"] = LGB_RANK_WEIGHT * combined["Rank"] + RULE_RANK_WEIGHT * combined["rule_rank"]
    combined = combined.sort_values(["Date", "combined_rank_score", "SecuritiesCode"], ascending=[True, True, True])
    combined["Rank"] = combined.groupby("Date").cumcount().astype("int32")
    prediction_frame = sample_prediction[["Date", "SecuritiesCode"]].merge(
        combined[["Date", "SecuritiesCode", "Rank"]],
        on=["Date", "SecuritiesCode"],
        how="left",
    )
    prediction_frame["Rank"] = prediction_frame["Rank"].astype("int32")
    env.predict(prediction_frame)
    counter += 1
