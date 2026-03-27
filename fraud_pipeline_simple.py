from pathlib import Path

import polars as pl

from data_cleaning import load_and_prepare, reduce_dataset, fit_scaler, apply_scaler
from data_analysis import perform_analysis, build_svg_confusion
from model_training import fit_logistic_regression, evaluate_logistic


def run_simple_pipeline(data_path: str = "data.csv") -> None:
    print("=== Pipeline simple de detection de fraude ===")
    Path("outputs").mkdir(exist_ok=True)

    df = load_and_prepare()
    print(f"Chargement: {df.height:,} lignes")

    reduced_df = reduce_dataset(df, 20_000)
    print(f"Reduction: {reduced_df.height:,} lignes")

    split_step = int(reduced_df.select(pl.col("step").quantile(0.75)).row(0)[0])
    train_df = reduced_df.filter(pl.col("step") <= split_step)
    test_df = reduced_df.filter(pl.col("step") > split_step)
    print(f"Train={train_df.height}, Test={test_df.height}")

    scaler = fit_scaler(train_df)
    train_df = apply_scaler(train_df, scaler)
    test_df = apply_scaler(test_df, scaler)

    model = fit_logistic_regression(train_df, [
        "step", "amount", "oldbalanceOrg", "oldbalanceDest", "amount_log", "oldbalanceOrg_log"
    ], epochs=50, learning_rate=0.2)

    test_confusion, test_scores = evaluate_logistic(model, test_df, threshold=0.5)
    print("Scores sur test:")
    for key, val in test_scores.items():
        print(f"  {key}: {val:.4f}")

    # Analyse + graphes
    analysis_results = perform_analysis(df, reduced_df)
    print("Analyse: classes", analysis_results["class_counts"])
    print("Analyse: fraudes par type", analysis_results["fraud_by_type"])

    build_svg_confusion(test_confusion, Path("outputs") / "confusion_matrix.svg")
    print("Graphes generes dans outputs/ (dont class_distribution.svg, fraud_by_type.svg, confusion_matrix.svg)")


if __name__ == "__main__":
    run_simple_pipeline()
