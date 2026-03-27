from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from data_cleaning import load_and_prepare, reduce_dataset, fit_scaler, apply_scaler
from data_analysis import perform_analysis, write_report, build_svg_confusion
from model_training import train_models

DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs")
REPORT_PATH = OUTPUT_DIR / "fraud_report.md"
TREE_PATH = OUTPUT_DIR / "fraud_tree.json"
LOGISTIC_PATH = OUTPUT_DIR / "fraud_logistic.json"
BAYES_PATH = OUTPUT_DIR / "fraud_gaussian_nb.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CLEANED_PATH = OUTPUT_DIR / "data_fraud_cleaned.parquet"
REDUCED_PATH = OUTPUT_DIR / "data_fraud_reduced_50000.parquet"
SCALED_REDUCED_PATH = OUTPUT_DIR / "data_fraud_reduced_50000_scaled.parquet"
CONFUSION_CSV_PATH = OUTPUT_DIR / "confusion_matrix.csv"
CLASS_SVG_PATH = OUTPUT_DIR / "class_distribution.svg"
TYPE_SVG_PATH = OUTPUT_DIR / "fraud_by_type.svg"
CONFUSION_SVG_PATH = OUTPUT_DIR / "confusion_matrix.svg"
TYPE_RATE_SVG_PATH = OUTPUT_DIR / "fraud_rate_by_type.svg"
STEP_RATE_SVG_PATH = OUTPUT_DIR / "fraud_rate_by_period.svg"
AMOUNT_RATE_SVG_PATH = OUTPUT_DIR / "fraud_rate_by_amount_bucket.svg"
SIGNAL_RATE_SVG_PATH = OUTPUT_DIR / "fraud_signal_comparison.svg"

ANALYSIS_BOOL_FEATURES = [
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_debit",
    "is_cash_in",
    "orig_zero",
    "dest_zero",
    "emptied_origin",
    "dest_untouched",
    "balance_error_orig",
    "balance_error_dest",
    "flagged_fraud",
]

ANALYSIS_NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "newbalanceOrig",
    "newbalanceDest",
    "amount_log",
    "oldbalanceOrg_log",
    "newbalanceOrig_log",
    "oldbalanceDest_log",
    "newbalanceDest_log",
    "amount_over_oldbalance",
    "dest_delta_log",
    "orig_delta_log",
]

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
PERCENTILES = [0.05, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95]
TARGET_ROWS = 50_000

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_and_prepare()
    reduced_df = reduce_dataset(df, TARGET_ROWS)

    split_train_max_step = int(reduced_df["step"].quantile(0.80, interpolation="nearest"))
    split_validation_max_step = int(reduced_df["step"].quantile(0.90, interpolation="nearest"))
    train_df_raw = reduced_df.filter(pl.col("step") <= split_train_max_step)
    validation_df_raw = reduced_df.filter(
        (pl.col("step") > split_train_max_step) & (pl.col("step") <= split_validation_max_step)
    )
    test_df_raw = reduced_df.filter(pl.col("step") > split_validation_max_step)

    scaling_stats = fit_scaler(train_df_raw)
    train_df = apply_scaler(train_df_raw, scaling_stats)
    validation_df = apply_scaler(validation_df_raw, scaling_stats)
    test_df = apply_scaler(test_df_raw, scaling_stats)
    reduced_scaled_df = apply_scaler(reduced_df, scaling_stats)
    reduced_scaled_df.write_parquet(SCALED_REDUCED_PATH)

    # Train models
    models_results = train_models(train_df, validation_df)

    # Perform analysis
    analysis_results = perform_analysis(df, reduced_df)

    # Generate confusion matrix SVG for tree
    build_svg_confusion(models_results["tree_test_confusion"], CONFUSION_SVG_PATH)

    # Write report
    write_report(
        df=df,
        reduced_df=reduced_df,
        class_counts=analysis_results["class_counts"],
        fraud_by_type=analysis_results["fraud_by_type"],
        tree_validation_confusion=models_results["tree_validation_confusion"],
        tree_validation_scores=models_results["tree_validation_scores"],
        tree_test_confusion=models_results["tree_test_confusion"],
        tree_test_scores=models_results["tree_test_scores"],
        tree_threshold=models_results["tree_threshold"],
        tree=models_results["tree"],
        logistic_validation_confusion=models_results["logistic_validation_confusion"],
        logistic_validation_scores=models_results["logistic_validation_scores"],
        logistic_test_confusion=models_results["logistic_test_confusion"],
        logistic_test_scores=models_results["logistic_test_scores"],
        logistic_threshold=models_results["logistic_threshold"],
        logistic_model=models_results["logistic_model"],
        bayes_validation_scores=models_results["bayes_validation_scores"],
        bayes_test_confusion=models_results["bayes_test_confusion"],
        bayes_test_scores=models_results["bayes_test_scores"],
        bayes_threshold=models_results["bayes_threshold"],
        train_rows=train_df_raw.height,
        validation_rows=validation_df_raw.height,
        test_rows=test_df_raw.height,
        train_frauds=int(train_df_raw["isFraud"].sum()),
        validation_frauds=int(validation_df_raw["isFraud"].sum()),
        test_frauds=int(test_df_raw["isFraud"].sum()),
        split_train_max_step=split_train_max_step,
        split_validation_max_step=split_validation_max_step,
        scaling_stats=scaling_stats,
    )

if __name__ == "__main__":
    main()
