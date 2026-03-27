from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from models import ScalingStat

DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs")
CLEANED_PATH = OUTPUT_DIR / "data_fraud_cleaned.parquet"
REDUCED_PATH = OUTPUT_DIR / "data_fraud_reduced_50000.parquet"
SCALED_REDUCED_PATH = OUTPUT_DIR / "data_fraud_reduced_50000_scaled.parquet"

MODEL_BOOL_FEATURES = [
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_debit",
    "is_cash_in",
    "orig_zero",
    "dest_zero",
]

MODEL_NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_log",
    "oldbalanceOrg_log",
]

MODEL_FEATURES = MODEL_BOOL_FEATURES + MODEL_NUMERIC_FEATURES
TARGET_ROWS = 50_000


def load_and_prepare() -> pl.DataFrame:
    df = (
        pl.read_csv(DATA_PATH)
        .with_row_index("row_id")
        .with_columns(
            [
                pl.col("type").cast(pl.Utf8),
                (pl.col("type") == "TRANSFER").cast(pl.Int8).alias("is_transfer"),
                (pl.col("type") == "CASH_OUT").cast(pl.Int8).alias("is_cash_out"),
                (pl.col("type") == "PAYMENT").cast(pl.Int8).alias("is_payment"),
                (pl.col("type") == "DEBIT").cast(pl.Int8).alias("is_debit"),
                (pl.col("type") == "CASH_IN").cast(pl.Int8).alias("is_cash_in"),
                (pl.col("oldbalanceOrg") == 0).cast(pl.Int8).alias("orig_zero"),
                (pl.col("oldbalanceDest") == 0).cast(pl.Int8).alias("dest_zero"),
                (
                    (pl.col("oldbalanceOrg") > 0) & (pl.col("newbalanceOrig") == 0)
                )
                .cast(pl.Int8)
                .alias("emptied_origin"),
                (
                    (pl.col("oldbalanceDest") == 0) & (pl.col("newbalanceDest") == 0)
                )
                .cast(pl.Int8)
                .alias("dest_untouched"),
                (
                    (pl.col("newbalanceOrig") + pl.col("amount") - pl.col("oldbalanceOrg"))
                    .abs()
                    > 1e-9
                )
                .cast(pl.Int8)
                .alias("balance_error_orig"),
                (
                    (pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest"))
                    .abs()
                    > 1e-9
                )
                .cast(pl.Int8)
                .alias("balance_error_dest"),
                pl.col("isFlaggedFraud").cast(pl.Int8).alias("flagged_fraud"),
                pl.col("amount").log1p().alias("amount_log"),
                pl.col("oldbalanceOrg").log1p().alias("oldbalanceOrg_log"),
                pl.col("newbalanceOrig").log1p().alias("newbalanceOrig_log"),
                pl.col("oldbalanceDest").log1p().alias("oldbalanceDest_log"),
                pl.col("newbalanceDest").log1p().alias("newbalanceDest_log"),
                (pl.col("amount") / (pl.col("oldbalanceOrg") + 1.0)).alias(
                    "amount_over_oldbalance"
                ),
                (pl.col("newbalanceDest") - pl.col("oldbalanceDest"))
                .abs()
                .log1p()
                .alias("dest_delta_log"),
                (pl.col("oldbalanceOrg") - pl.col("newbalanceOrig"))
                .abs()
                .log1p()
                .alias("orig_delta_log"),
            ]
        )
        .drop(["nameOrig", "nameDest"])
    )
    df.write_parquet(CLEANED_PATH)
    return df


def reduce_dataset(df: pl.DataFrame, target_rows: int = TARGET_ROWS) -> pl.DataFrame:
    fraud_df = df.filter(pl.col("isFraud") == 1)
    nonfraud_df = (
        df.filter(pl.col("isFraud") == 0)
        .with_columns(
            (
                (pl.col("row_id").cast(pl.Int64) * 1103515245 + 12345) % 2147483647
            ).alias("sample_key")
        )
        .sort("sample_key")
    )
    nonfraud_target = max(0, target_rows - fraud_df.height)
    reduced = (
        pl.concat([fraud_df, nonfraud_df.head(nonfraud_target).drop("sample_key")])
        .sort(["step", "row_id"])
    )
    reduced.write_parquet(REDUCED_PATH)
    return reduced


def fit_scaler(train_df: pl.DataFrame) -> dict[str, ScalingStat]:
    stats: dict[str, ScalingStat] = {}
    for feature in MODEL_NUMERIC_FEATURES:
        row = train_df.select(
            [
                pl.col(feature).quantile(0.01).alias("clip_low"),
                pl.col(feature).quantile(0.99).alias("clip_high"),
            ]
        ).row(0, named=True)
        clip_low = float(row["clip_low"])
        clip_high = float(row["clip_high"])
        clipped = train_df.select(
            pl.col(feature).clip(clip_low, clip_high).alias(feature)
        )
        mean_std = clipped.select(
            [
                pl.col(feature).mean().alias("mean"),
                pl.col(feature).std().alias("std"),
            ]
        ).row(0, named=True)
        mean = float(mean_std["mean"])
        std = float(mean_std["std"]) if mean_std["std"] not in (None, 0.0) else 1.0
        stats[feature] = ScalingStat(
            feature=feature,
            clip_low=clip_low,
            clip_high=clip_high,
            mean=mean,
            std=std,
        )
    return stats


def apply_scaler(df: pl.DataFrame, stats: dict[str, ScalingStat]) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for feature in MODEL_NUMERIC_FEATURES:
        stat = stats[feature]
        expressions.append(
            (
                (pl.col(feature).clip(stat.clip_low, stat.clip_high) - stat.mean) / stat.std
            ).alias(feature)
        )
    return df.with_columns(expressions)