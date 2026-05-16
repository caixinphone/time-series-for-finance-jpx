from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = ""
except (ImportError, OSError) as exc:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = str(exc)


PRICE_COLUMNS = [
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


@dataclass(frozen=True)
class TrainResult:
    model: object
    feature_columns: list[str]
    validation_score: float
    validation_rmse: float
    validation_dates: tuple[str, str]
    model_name: str


def read_prices(path: Path, *, include_target: bool = True) -> pd.DataFrame:
    columns = PRICE_COLUMNS if include_target else [c for c in PRICE_COLUMNS if c != "Target"]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in columns if c in available]
    prices = pd.read_csv(path, usecols=usecols, parse_dates=["Date"])
    prices["SecuritiesCode"] = prices["SecuritiesCode"].astype("int32")
    return prices


def read_stock_list(path: Path) -> pd.DataFrame:
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


def add_adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["SecuritiesCode", "Date"]).copy()
    factor = prices["AdjustmentFactor"].fillna(1.0).replace(0, 1.0)
    prices["_reverse_factor"] = factor
    prices["_cum_adjustment"] = (
        prices.sort_values(["SecuritiesCode", "Date"], ascending=[True, False])
        .groupby("SecuritiesCode", sort=False)["_reverse_factor"]
        .cumprod()
    )
    prices["AdjustedClose"] = prices["Close"] * prices["_cum_adjustment"]
    prices = prices.drop(columns=["_reverse_factor", "_cum_adjustment"])
    return prices.sort_values(["SecuritiesCode", "Date"])


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = add_adjusted_close(prices)
    group = prices.groupby("SecuritiesCode", sort=False)

    prices["ret_1"] = group["AdjustedClose"].pct_change(1)
    for window in [2, 5, 10, 20]:
        prices[f"ret_{window}"] = group["AdjustedClose"].pct_change(window)

    for window in [5, 10, 20]:
        shifted_ret = group["ret_1"].shift(1)
        prices[f"ret_mean_{window}"] = (
            shifted_ret.groupby(prices["SecuritiesCode"], sort=False)
            .rolling(window, min_periods=max(2, window // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        prices[f"ret_std_{window}"] = (
            shifted_ret.groupby(prices["SecuritiesCode"], sort=False)
            .rolling(window, min_periods=max(2, window // 2))
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


def make_features(prices: pd.DataFrame, stock_list: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = add_price_features(prices)
    features = features.merge(stock_list, on="SecuritiesCode", how="left")
    features["market_cap_log"] = np.log1p(features["MarketCapitalization"])
    features["issued_shares_log"] = np.log1p(features["IssuedShares"])

    feature_columns = [
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
    features[feature_columns] = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    return features, feature_columns


def split_by_last_dates(df: pd.DataFrame, valid_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(df["Date"].dropna().unique()))
    if valid_days <= 0 or valid_days >= len(dates):
        raise ValueError(f"valid_days must be between 1 and {len(dates) - 1}")
    valid_dates = set(dates[-valid_days:])
    valid_mask = df["Date"].isin(valid_dates)
    return df.loc[~valid_mask].copy(), df.loc[valid_mask].copy()


def add_ranks(df: pd.DataFrame, pred_col: str = "prediction") -> pd.DataFrame:
    ranked = df.sort_values(["Date", pred_col, "SecuritiesCode"], ascending=[True, False, True]).copy()
    ranked["Rank"] = ranked.groupby("Date").cumcount().astype("int32")
    return ranked.sort_index()


def daily_spread_returns(
    df: pd.DataFrame,
    *,
    target_col: str = "Target",
    rank_col: str = "Rank",
    portfolio_size: int = 200,
    top_rank_weight_ratio: float = 2.0,
) -> pd.Series:
    daily_returns = {}
    for date, group in df.groupby("Date"):
        n = len(group)
        size = min(portfolio_size, n // 2)
        if size == 0:
            continue
        weights = np.linspace(top_rank_weight_ratio, 1.0, size)
        top = group.nsmallest(size, rank_col).sort_values(rank_col)
        bottom = group.nlargest(size, rank_col).sort_values(rank_col, ascending=False)
        long_return = (top[target_col].to_numpy() * weights).sum() / weights.mean()
        short_return = (bottom[target_col].to_numpy() * weights).sum() / weights.mean()
        daily_returns[pd.Timestamp(date)] = long_return - short_return
    return pd.Series(daily_returns).sort_index()


def sharpe_ratio(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(returns.mean() / std)


def train_model(
    features: pd.DataFrame,
    feature_columns: list[str],
    *,
    valid_days: int,
    random_state: int,
    max_train_rows: int | None,
) -> TrainResult:
    trainable = features.dropna(subset=["Target"]).copy()
    train_df, valid_df = split_by_last_dates(trainable, valid_days)
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=random_state).sort_values(["Date", "SecuritiesCode"])

    x_train = train_df[feature_columns]
    y_train = train_df["Target"]
    x_valid = valid_df[feature_columns]
    y_valid = valid_df["Target"]
    model, model_name = make_regressor(random_state=random_state, final=False)
    if LIGHTGBM_AVAILABLE:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100)],
        )
    else:
        model.fit(x_train, y_train)

    valid_pred = valid_df[["Date", "SecuritiesCode", "Target"]].copy()
    valid_pred["prediction"] = model.predict(x_valid)
    valid_pred = add_ranks(valid_pred)
    daily_returns = daily_spread_returns(valid_pred)
    validation_score = sharpe_ratio(daily_returns)
    validation_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred["prediction"])))

    validation_dates = (
        pd.Timestamp(valid_df["Date"].min()).strftime("%Y-%m-%d"),
        pd.Timestamp(valid_df["Date"].max()).strftime("%Y-%m-%d"),
    )
    return TrainResult(model, feature_columns, validation_score, validation_rmse, validation_dates, model_name)


def make_regressor(*, random_state: int, final: bool) -> tuple[object, str]:
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=400 if final else 600,
            learning_rate=0.035 if final else 0.03,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        return model, "LightGBMRegressor"

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=260 if final else 350,
        max_leaf_nodes=63,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
        verbose=0,
    )
    return model, "HistGradientBoostingRegressor"


def fit_final_model(
    features: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    max_train_rows: int | None,
) -> object:
    trainable = features.dropna(subset=["Target"]).copy()
    if max_train_rows and len(trainable) > max_train_rows:
        trainable = trainable.sample(max_train_rows, random_state=random_state).sort_values(["Date", "SecuritiesCode"])
    model, _ = make_regressor(random_state=random_state, final=True)
    model.fit(trainable[feature_columns], trainable["Target"])
    return model


def build_submission(
    data_root: Path,
    stock_list: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    output_path: Path,
) -> pd.DataFrame:
    train_prices = read_prices(data_root / "train_files" / "stock_prices.csv", include_target=True)
    sample = pd.read_csv(data_root / "example_test_files" / "sample_submission.csv", parse_dates=["Date"])
    sample_dates = set(sample["Date"].unique())

    test_parts = [read_prices(data_root / "example_test_files" / "stock_prices.csv", include_target=False)]
    supplemental_path = data_root / "supplemental_files" / "stock_prices.csv"
    if supplemental_path.exists():
        supplemental = read_prices(supplemental_path, include_target=True).drop(columns=["Target"])
        test_parts.append(supplemental)

    test_prices = (
        pd.concat(test_parts, ignore_index=True, sort=False)
        .sort_values(["Date", "SecuritiesCode"])
        .drop_duplicates(["Date", "SecuritiesCode"], keep="first")
    )
    prediction_prices = test_prices.loc[test_prices["Date"].isin(sample_dates)].copy()
    history = pd.concat([train_prices.drop(columns=["Target"]), prediction_prices], ignore_index=True, sort=False)
    history_features, _ = make_features(history, stock_list)

    test_features = history_features.loc[history_features["Date"].isin(sample_dates)].copy()
    test_features["prediction"] = model.predict(test_features[feature_columns])
    ranked = add_ranks(test_features[["Date", "SecuritiesCode", "prediction"]])
    submission = ranked[["Date", "SecuritiesCode", "Rank"]].copy()

    submission = sample[["Date", "SecuritiesCode"]].merge(submission, on=["Date", "SecuritiesCode"], how="left")
    if submission["Rank"].isna().any():
        missing_dates = submission.loc[submission["Rank"].isna(), "Date"].dt.strftime("%Y-%m-%d").unique()[:10]
        raise RuntimeError(f"Some sample submission rows did not receive a rank. Missing dates include {missing_dates}")
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission["Rank"] = submission["Rank"].astype("int32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def validate_submission(submission: pd.DataFrame) -> dict[str, object]:
    expected_columns = ["Date", "SecuritiesCode", "Rank"]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"Submission columns must be {expected_columns}")
    checks = {
        "rows": int(len(submission)),
        "dates": int(submission["Date"].nunique()),
        "securities": int(submission["SecuritiesCode"].nunique()),
        "rank_nulls": int(submission["Rank"].isna().sum()),
        "rank_unique_per_date": True,
        "rank_zero_based_per_date": True,
    }
    for _, group in submission.groupby("Date"):
        ranks = sorted(group["Rank"].tolist())
        if len(ranks) != len(set(ranks)):
            checks["rank_unique_per_date"] = False
        if ranks != list(range(len(ranks))):
            checks["rank_zero_based_per_date"] = False
    if not checks["rank_unique_per_date"] or not checks["rank_zero_based_per_date"]:
        raise ValueError(f"Rank validation failed: {checks}")
    return checks


def write_outputs(
    output_dir: Path,
    result: TrainResult,
    model: object,
    feature_columns: Iterable[str],
    submission_checks: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    importances = getattr(model, "feature_importances_", np.zeros(len(list(feature_columns)), dtype=float))
    importance = pd.DataFrame({"feature": list(feature_columns), "importance": importances}).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    metrics = {
        "validation_sharpe": result.validation_score,
        "validation_rmse": result.validation_rmse,
        "validation_start": result.validation_dates[0],
        "validation_end": result.validation_dates[1],
        "model": result.model_name,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "lightgbm_import_error": "" if LIGHTGBM_AVAILABLE else LIGHTGBM_IMPORT_ERROR,
        "submission_checks": submission_checks,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JPX Tokyo Stock Exchange Prediction baseline")
    parser.add_argument("--data-root", type=Path, default=Path("data/jpx"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--valid-days", type=int, default=60)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional sampling cap for faster local experiments. Defaults to full training data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_root / "train_files" / "stock_prices.csv"
    stock_list_path = args.data_root / "stock_list.csv"
    if not train_path.exists() or not stock_list_path.exists():
        raise FileNotFoundError(
            "JPX data not found. Expected data/jpx/train_files/stock_prices.csv and data/jpx/stock_list.csv"
        )
    if LIGHTGBM_AVAILABLE:
        print("Model backend: LightGBM")
    else:
        print("Model backend: sklearn HistGradientBoostingRegressor")
        print("LightGBM unavailable; reason:", LIGHTGBM_IMPORT_ERROR.splitlines()[0])

    print("Loading data...")
    prices = read_prices(train_path, include_target=True)
    stock_list = read_stock_list(stock_list_path)

    print("Building features...")
    features, feature_columns = make_features(prices, stock_list)

    print("Training validation model...")
    result = train_model(
        features,
        feature_columns,
        valid_days=args.valid_days,
        random_state=args.random_state,
        max_train_rows=args.max_train_rows,
    )
    print(f"Validation dates: {result.validation_dates[0]} to {result.validation_dates[1]}")
    print(f"Validation Sharpe: {result.validation_score:.6f}")
    print(f"Validation RMSE: {result.validation_rmse:.6f}")

    print("Training final model on all labeled training rows...")
    final_model = fit_final_model(
        features,
        feature_columns,
        random_state=args.random_state,
        max_train_rows=args.max_train_rows,
    )

    print("Building example-test submission...")
    submission = build_submission(args.data_root, stock_list, final_model, feature_columns, args.output)
    checks = validate_submission(submission)
    write_outputs(args.output.parent, result, final_model, feature_columns, checks)
    print(f"Wrote submission: {args.output}")
    print(f"Submission checks: {checks}")


if __name__ == "__main__":
    main()
