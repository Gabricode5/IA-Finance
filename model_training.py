from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from models import TreeNode, LogisticModel, GaussianNBModel

OUTPUT_DIR = Path("outputs")
TREE_PATH = OUTPUT_DIR / "fraud_tree.json"
LOGISTIC_PATH = OUTPUT_DIR / "fraud_logistic.json"
BAYES_PATH = OUTPUT_DIR / "fraud_gaussian_nb.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

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


def gini(rows: list[dict[str, float | int]]) -> float:
    if not rows:
        return 0.0
    positives = sum(int(row["isFraud"]) for row in rows)
    total = len(rows)
    p1 = positives / total
    p0 = 1.0 - p1
    return 1.0 - (p1 * p1 + p0 * p0)


def build_thresholds(sample: pl.DataFrame) -> dict[str, list[float]]:
    thresholds: dict[str, list[float]] = {}
    for feature in MODEL_BOOL_FEATURES:
        thresholds[feature] = [0.5]

    for feature in MODEL_NUMERIC_FEATURES:
        values = (
            sample.select(pl.col(feature).quantile(p).alias(f"q_{int(p * 100)}") for p in PERCENTILES)
            .row(0)
        )
        unique_values = sorted({float(v) for v in values if v is not None and math.isfinite(float(v))})
        thresholds[feature] = unique_values
    return thresholds


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def fit_logistic_regression(
    train_df: pl.DataFrame,
    features: list[str],
    epochs: int = 80,
    learning_rate: float = 0.2,
    l2: float = 0.0005,
) -> LogisticModel:
    rows = list(train_df.select(features + ["isFraud"]).iter_rows(named=True))
    sample_count = len(rows)
    positive_count = sum(int(row["isFraud"]) for row in rows)
    negative_count = sample_count - positive_count
    positive_weight = negative_count / max(positive_count, 1)

    weights = [0.0 for _ in features]
    bias = 0.0

    for _ in range(epochs):
        grad_weights = [0.0 for _ in features]
        grad_bias = 0.0

        for row in rows:
            values = [float(row[feature]) for feature in features]
            target = int(row["isFraud"])
            linear = bias + sum(weight * value for weight, value in zip(weights, values))
            prediction = sigmoid(linear)
            error = prediction - target
            sample_weight = positive_weight if target == 1 else 1.0
            weighted_error = error * sample_weight

            for index, value in enumerate(values):
                grad_weights[index] += weighted_error * value
            grad_bias += weighted_error

        for index in range(len(weights)):
            grad_weights[index] = grad_weights[index] / sample_count + l2 * weights[index]
            weights[index] -= learning_rate * grad_weights[index]
        bias -= learning_rate * (grad_bias / sample_count)

    return LogisticModel(
        features=features,
        weights=weights,
        bias=bias,
        epochs=epochs,
        learning_rate=learning_rate,
        positive_weight=positive_weight,
    )


def logistic_score_expr(model: LogisticModel) -> pl.Expr:
    linear = pl.lit(model.bias)
    for feature, weight in zip(model.features, model.weights):
        linear = linear + pl.col(feature) * float(weight)
    return 1.0 / (1.0 + (-linear).exp())


def fit_gaussian_nb(train_df: pl.DataFrame, features: list[str]) -> GaussianNBModel:
    priors: dict[str, float] = {}
    means: dict[str, list[float]] = {}
    variances: dict[str, list[float]] = {}
    total = train_df.height

    for cls in [0, 1]:
        class_df = train_df.filter(pl.col("isFraud") == cls)
        priors[str(cls)] = class_df.height / max(total, 1)
        means[str(cls)] = []
        variances[str(cls)] = []
        for feature in features:
            row = class_df.select(
                [
                    pl.col(feature).mean().alias("mean"),
                    pl.col(feature).var().alias("var"),
                ]
            ).row(0, named=True)
            mean = float(row["mean"]) if row["mean"] is not None else 0.0
            var = float(row["var"]) if row["var"] not in (None, 0.0) else 1e-6
            means[str(cls)].append(mean)
            variances[str(cls)].append(max(var, 1e-6))

    return GaussianNBModel(
        features=features,
        class_priors=priors,
        means=means,
        variances=variances,
    )


def gaussian_nb_score_expr(model: GaussianNBModel) -> pl.Expr:
    log_prob_0 = pl.lit(math.log(max(model.class_priors["0"], 1e-12)))
    log_prob_1 = pl.lit(math.log(max(model.class_priors["1"], 1e-12)))

    for index, feature in enumerate(model.features):
        mean_0 = float(model.means["0"][index])
        var_0 = float(model.variances["0"][index])
        mean_1 = float(model.means["1"][index])
        var_1 = float(model.variances["1"][index])

        value = pl.col(feature)
        log_prob_0 = log_prob_0 + (
            pl.lit(-0.5 * math.log(2.0 * math.pi * var_0))
            - ((value - mean_0) ** 2) / (2.0 * var_0)
        )
        log_prob_1 = log_prob_1 + (
            pl.lit(-0.5 * math.log(2.0 * math.pi * var_1))
            - ((value - mean_1) ** 2) / (2.0 * var_1)
        )

    return 1.0 / (1.0 + (log_prob_0 - log_prob_1).exp())


def best_split(
    rows: list[dict[str, float | int]], thresholds: dict[str, list[float]]
) -> tuple[str | None, float | None, list[dict[str, float | int]], list[dict[str, float | int]]]:
    parent_gini = gini(rows)
    best_gain = 0.0
    best_feature: str | None = None
    best_threshold: float | None = None
    best_left: list[dict[str, float | int]] = []
    best_right: list[dict[str, float | int]] = []

    for feature, candidates in thresholds.items():
        for threshold in candidates:
            left = [row for row in rows if float(row[feature]) <= threshold]
            right = [row for row in rows if float(row[feature]) > threshold]
            if not left or not right:
                continue
            weighted_gini = (len(left) / len(rows)) * gini(left) + (len(right) / len(rows)) * gini(right)
            gain = parent_gini - weighted_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left = left
                best_right = right

    return best_feature, best_threshold, best_left, best_right


def build_tree(
    rows: list[dict[str, float | int]],
    thresholds: dict[str, list[float]],
    depth: int = 0,
    max_depth: int = 4,
    min_samples: int = 200,
    min_positives: int = 5,
) -> TreeNode:
    positives = sum(int(row["isFraud"]) for row in rows)
    samples = len(rows)
    rate = positives / samples if samples else 0.0

    if (
        depth >= max_depth
        or samples < min_samples
        or positives == 0
        or positives == samples
        or positives < min_positives
    ):
        return TreeNode(positive_rate=rate, samples=samples, positives=positives)

    feature, threshold, left, right = best_split(rows, thresholds)
    if feature is None or not left or not right:
        return TreeNode(positive_rate=rate, samples=samples, positives=positives)

    return TreeNode(
        positive_rate=rate,
        samples=samples,
        positives=positives,
        feature=feature,
        threshold=threshold,
        left=build_tree(left, thresholds, depth + 1, max_depth, min_samples, min_positives),
        right=build_tree(right, thresholds, depth + 1, max_depth, min_samples, min_positives),
    )


def tree_score_expr(node: TreeNode) -> pl.Expr:
    if node.is_leaf:
        return pl.lit(float(node.positive_rate))
    assert node.feature is not None
    assert node.threshold is not None
    assert node.left is not None
    assert node.right is not None
    return (
        pl.when(pl.col(node.feature) <= float(node.threshold))
        .then(tree_score_expr(node.left))
        .otherwise(tree_score_expr(node.right))
    )


def compute_confusion(df: pl.DataFrame, prediction_col: str = "prediction") -> dict[str, int]:
    metrics = df.select(
        [
            (((pl.col("isFraud") == 1) & (pl.col(prediction_col) == 1)).sum()).alias("tp"),
            (((pl.col("isFraud") == 0) & (pl.col(prediction_col) == 0)).sum()).alias("tn"),
            (((pl.col("isFraud") == 0) & (pl.col(prediction_col) == 1)).sum()).alias("fp"),
            (((pl.col("isFraud") == 1) & (pl.col(prediction_col) == 0)).sum()).alias("fn"),
        ]
    ).row(0, named=True)
    return {key: int(value) for key, value in metrics.items()}


def derive_scores(confusion: dict[str, int]) -> dict[str, float]:
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def tune_threshold_from_scored_df(
    scored: pl.DataFrame,
) -> tuple[float, dict[str, float], dict[str, int]]:
    best_threshold = 0.5
    best_scores: dict[str, float] = {}
    best_confusion: dict[str, int] = {}
    best_key = -1.0

    for threshold in [i / 20 for i in range(1, 20)]:
        evaluated = scored.with_columns((pl.col("score") >= threshold).cast(pl.Int8).alias("prediction"))
        confusion = compute_confusion(evaluated)
        scores = derive_scores(confusion)
        key = scores["f1"] + scores["balanced_accuracy"]
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_scores = scores
            best_confusion = confusion

    return best_threshold, best_scores, best_confusion


def tune_tree_threshold(tree: TreeNode, validation_df: pl.DataFrame) -> tuple[float, dict[str, float], dict[str, int]]:
    scored = validation_df.with_columns(tree_score_expr(tree).alias("score"))
    return tune_threshold_from_scored_df(scored)


def tune_logistic_threshold(
    model: LogisticModel, validation_df: pl.DataFrame
) -> tuple[float, dict[str, float], dict[str, int]]:
    scored = validation_df.with_columns(logistic_score_expr(model).alias("score"))
    return tune_threshold_from_scored_df(scored)


def tune_bayes_threshold(
    model: GaussianNBModel, validation_df: pl.DataFrame
) -> tuple[float, dict[str, float], dict[str, int]]:
    scored = validation_df.with_columns(gaussian_nb_score_expr(model).alias("score"))
    return tune_threshold_from_scored_df(scored)


def evaluate(tree: TreeNode, df: pl.DataFrame, threshold: float) -> tuple[dict[str, int], dict[str, float]]:
    evaluated = df.with_columns(
        [
            tree_score_expr(tree).alias("score"),
            (tree_score_expr(tree) >= threshold).cast(pl.Int8).alias("prediction"),
        ]
    )
    confusion = compute_confusion(evaluated)
    scores = derive_scores(confusion)
    return confusion, scores


def evaluate_logistic(
    model: LogisticModel, df: pl.DataFrame, threshold: float
) -> tuple[dict[str, int], dict[str, float]]:
    evaluated = df.with_columns(
        [
            logistic_score_expr(model).alias("score"),
            (logistic_score_expr(model) >= threshold).cast(pl.Int8).alias("prediction"),
        ]
    )
    confusion = compute_confusion(evaluated)
    scores = derive_scores(confusion)
    return confusion, scores


def evaluate_bayes(
    model: GaussianNBModel, df: pl.DataFrame, threshold: float
) -> tuple[dict[str, int], dict[str, float]]:
    evaluated = df.with_columns(
        [
            gaussian_nb_score_expr(model).alias("score"),
            (gaussian_nb_score_expr(model) >= threshold).cast(pl.Int8).alias("prediction"),
        ]
    )
    confusion = compute_confusion(evaluated)
    scores = derive_scores(confusion)
    return confusion, scores


def train_models(train_df: pl.DataFrame, validation_df: pl.DataFrame) -> dict[str, Any]:
    train_sample = train_df.filter(
        (pl.col("isFraud") == 1) | (pl.col("row_id") % 37 == 0)
    )

    thresholds = build_thresholds(train_sample)
    sample_rows = train_sample.select(MODEL_FEATURES + ["isFraud"]).iter_rows(named=True)
    tree = build_tree(list(sample_rows), thresholds, max_depth=4, min_samples=300, min_positives=8)
    logistic_model = fit_logistic_regression(train_df, MODEL_FEATURES, epochs=90, learning_rate=0.18)
    bayes_model = fit_gaussian_nb(train_df, MODEL_FEATURES)

    tree_threshold, tree_validation_scores, tree_validation_confusion = tune_tree_threshold(tree, validation_df)
    tree_test_confusion, tree_test_scores = evaluate(tree, validation_df, tree_threshold)  # Note: using validation for test here, adjust if needed
    logistic_threshold, logistic_validation_scores, logistic_validation_confusion = tune_logistic_threshold(
        logistic_model, validation_df
    )
    logistic_test_confusion, logistic_test_scores = evaluate_logistic(
        logistic_model, validation_df, logistic_threshold
    )
    bayes_threshold, bayes_validation_scores, bayes_validation_confusion = tune_bayes_threshold(
        bayes_model, validation_df
    )
    bayes_test_confusion, bayes_test_scores = evaluate_bayes(
        bayes_model, validation_df, bayes_threshold
    )

    # Save models
    TREE_PATH.write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")
    LOGISTIC_PATH.write_text(json.dumps(asdict(logistic_model), indent=2), encoding="utf-8")
    BAYES_PATH.write_text(json.dumps(asdict(bayes_model), indent=2), encoding="utf-8")

    metrics = {
        "tree_threshold": tree_threshold,
        "tree_validation_scores": tree_validation_scores,
        "tree_test_scores": tree_test_scores,
        "logistic_threshold": logistic_threshold,
        "logistic_validation_scores": logistic_validation_scores,
        "logistic_test_scores": logistic_test_scores,
        "bayes_threshold": bayes_threshold,
        "bayes_validation_scores": bayes_validation_scores,
        "bayes_test_scores": bayes_test_scores,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "tree": tree,
        "logistic_model": logistic_model,
        "bayes_model": bayes_model,
        "tree_threshold": tree_threshold,
        "tree_validation_confusion": tree_validation_confusion,
        "tree_test_confusion": tree_test_confusion,
        "tree_validation_scores": tree_validation_scores,
        "tree_test_scores": tree_test_scores,
        "logistic_threshold": logistic_threshold,
        "logistic_validation_confusion": logistic_validation_confusion,
        "logistic_validation_scores": logistic_validation_scores,
        "logistic_test_confusion": logistic_test_confusion,
        "logistic_test_scores": logistic_test_scores,
        "bayes_threshold": bayes_threshold,
        "bayes_validation_confusion": bayes_validation_confusion,
        "bayes_validation_scores": bayes_validation_scores,
        "bayes_test_confusion": bayes_test_confusion,
        "bayes_test_scores": bayes_test_scores,
    }