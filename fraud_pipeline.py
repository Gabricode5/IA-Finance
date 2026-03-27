from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl


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


@dataclass
class TreeNode:
    positive_rate: float
    samples: int
    positives: int
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.left is not None:
            data["left"] = self.left.to_dict()
        if self.right is not None:
            data["right"] = self.right.to_dict()
        return data


@dataclass
class ScalingStat:
    feature: str
    clip_low: float
    clip_high: float
    mean: float
    std: float


@dataclass
class LogisticModel:
    features: list[str]
    weights: list[float]
    bias: float
    epochs: int
    learning_rate: float
    positive_weight: float


@dataclass
class GaussianNBModel:
    features: list[str]
    class_priors: dict[str, float]
    means: dict[str, list[float]]
    variances: dict[str, list[float]]


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


def tune_bayes_threshold(
    model: GaussianNBModel, validation_df: pl.DataFrame
) -> tuple[float, dict[str, float], dict[str, int]]:
    scored = validation_df.with_columns(gaussian_nb_score_expr(model).alias("score"))
    return tune_threshold_from_scored_df(scored)


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


def build_svg_bar_chart(
    items: list[tuple[str, float]],
    path: Path,
    title: str,
    color: str,
    value_format: str = "int",
) -> None:
    width = 900
    height = 420
    left = 100
    bottom = 70
    top = 70
    bar_gap = 40
    bar_width = max(60, int((width - left - 80 - bar_gap * (len(items) - 1)) / max(len(items), 1)))
    chart_height = height - top - bottom
    max_value = max(value for _, value in items) if items else 1.0
    bars = []
    labels = []
    values = []
    x = left
    for label, value in items:
        bar_height = 0 if max_value == 0 else (value / max_value) * chart_height
        y = height - bottom - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{color}" rx="6" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - 30}" font-size="15" text-anchor="middle">{label}</text>'
        )
        text_value = f"{int(value):,}".replace(",", " ") if value_format == "int" else f"{value:.2%}"
        values.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" font-size="14" text-anchor="middle">{text_value}</text>'
        )
        x += bar_width + bar_gap

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="38" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - 30}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
{''.join(bars)}
{''.join(labels)}
{''.join(values)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_svg_grouped_chart(
    groups: list[str],
    normal_values: list[int],
    fraud_values: list[int],
    path: Path,
    title: str,
) -> None:
    width = 1000
    height = 500
    left = 90
    bottom = 90
    top = 70
    chart_height = height - top - bottom
    group_width = 140
    bar_width = 40
    gap = 18
    max_value = max(normal_values + fraud_values) if groups else 1
    elements: list[str] = []
    x = left

    for group, normal, fraud in zip(groups, normal_values, fraud_values):
        n_height = (normal / max_value) * chart_height if max_value else 0
        f_height = (fraud / max_value) * chart_height if max_value else 0
        n_y = height - bottom - n_height
        f_y = height - bottom - f_height
        elements.append(
            f'<rect x="{x}" y="{n_y:.2f}" width="{bar_width}" height="{n_height:.2f}" fill="#5b8c5a" rx="5" />'
        )
        elements.append(
            f'<rect x="{x + bar_width + gap}" y="{f_y:.2f}" width="{bar_width}" height="{f_height:.2f}" fill="#c8553d" rx="5" />'
        )
        elements.append(
            f'<text x="{x + group_width / 2}" y="{height - 35}" font-size="13" text-anchor="middle">{group}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2}" y="{n_y - 10:.2f}" font-size="11" text-anchor="middle">{normal:,}</text>'.replace(",", " ")
        )
        elements.append(
            f'<text x="{x + bar_width + gap + bar_width / 2}" y="{f_y - 10:.2f}" font-size="11" text-anchor="middle">{fraud:,}</text>'.replace(",", " ")
        )
        x += group_width + 30

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="38" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - 30}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<rect x="{width - 210}" y="22" width="18" height="18" fill="#5b8c5a" rx="4" />
<text x="{width - 186}" y="36" font-size="13">Normal</text>
<rect x="{width - 120}" y="22" width="18" height="18" fill="#c8553d" rx="4" />
<text x="{width - 96}" y="36" font-size="13">Fraude</text>
{''.join(elements)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_svg_grouped_rate_chart(
    groups: list[str],
    normal_values: list[float],
    fraud_values: list[float],
    path: Path,
    title: str,
    normal_label: str = "Normal",
    fraud_label: str = "Fraude",
) -> None:
    width = max(1000, 160 + len(groups) * 120)
    height = 500
    left = 90
    bottom = 100
    top = 70
    chart_height = height - top - bottom
    group_width = 90
    bar_width = 28
    gap = 12
    max_value = max(normal_values + fraud_values) if groups else 1.0
    elements: list[str] = []
    x = left

    for group, normal, fraud in zip(groups, normal_values, fraud_values):
        n_height = (normal / max_value) * chart_height if max_value else 0.0
        f_height = (fraud / max_value) * chart_height if max_value else 0.0
        n_y = height - bottom - n_height
        f_y = height - bottom - f_height
        elements.append(
            f'<rect x="{x}" y="{n_y:.2f}" width="{bar_width}" height="{n_height:.2f}" fill="#5b8c5a" rx="5" />'
        )
        elements.append(
            f'<rect x="{x + bar_width + gap}" y="{f_y:.2f}" width="{bar_width}" height="{f_height:.2f}" fill="#c8553d" rx="5" />'
        )
        elements.append(
            f'<text x="{x + group_width / 2}" y="{height - 40}" font-size="12" text-anchor="middle">{group}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2}" y="{n_y - 10:.2f}" font-size="11" text-anchor="middle">{normal:.1%}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width + gap + bar_width / 2}" y="{f_y - 10:.2f}" font-size="11" text-anchor="middle">{fraud:.1%}</text>'
        )
        x += group_width + 24

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="38" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>
<line x1="{left}" y1="{height - bottom}" x2="{width - 30}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444" stroke-width="2"/>
<rect x="{width - 240}" y="22" width="18" height="18" fill="#5b8c5a" rx="4" />
<text x="{width - 216}" y="36" font-size="13">{normal_label}</text>
<rect x="{width - 120}" y="22" width="18" height="18" fill="#c8553d" rx="4" />
<text x="{width - 96}" y="36" font-size="13">{fraud_label}</text>
{''.join(elements)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_svg_rate_chart(
    items: list[tuple[str, float]],
    path: Path,
    title: str,
    color: str,
) -> None:
    build_svg_bar_chart(items, path, title, color, value_format="percent")


def build_svg_confusion(confusion: dict[str, int], path: Path) -> None:
    width = 620
    height = 500
    x0 = 170
    y0 = 120
    cell = 130
    cells = [
        ("TN", confusion["tn"], x0, y0, "#d9ead3"),
        ("FP", confusion["fp"], x0 + cell, y0, "#f4cccc"),
        ("FN", confusion["fn"], x0, y0 + cell, "#fce5cd"),
        ("TP", confusion["tp"], x0 + cell, y0 + cell, "#cfe2f3"),
    ]
    cell_svg = []
    for label, value, x, y, fill in cells:
        cell_svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#444" stroke-width="2"/>')
        cell_svg.append(f'<text x="{x + cell / 2}" y="{y + 48}" font-size="22" text-anchor="middle" font-weight="bold">{label}</text>')
        cell_svg.append(
            f'<text x="{x + cell / 2}" y="{y + 84}" font-size="18" text-anchor="middle">{value:,}</text>'.replace(",", " ")
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fcfbf7"/>
<text x="{width / 2}" y="46" font-size="24" text-anchor="middle" font-weight="bold">Matrice de confusion</text>
<text x="{x0 + cell}" y="92" font-size="18" text-anchor="middle">Pred 0</text>
<text x="{x0 + 2 * cell}" y="92" font-size="18" text-anchor="middle">Pred 1</text>
<text x="82" y="{y0 + 72}" font-size="18" text-anchor="middle">Vrai 0</text>
<text x="82" y="{y0 + 202}" font-size="18" text-anchor="middle">Vrai 1</text>
{''.join(cell_svg)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def render_tree(node: TreeNode, depth: int = 0) -> list[str]:
    indent = "  " * depth
    if node.is_leaf:
        return [
            f"{indent}- Feuille: taux_fraude={node.positive_rate:.4f}, echantillons={node.samples}, fraudes={node.positives}"
        ]

    assert node.feature is not None
    assert node.threshold is not None
    assert node.left is not None
    assert node.right is not None
    lines = [
        f"{indent}- Si `{node.feature} <= {node.threshold:.4f}` (n={node.samples}, fraude={node.positives})"
    ]
    lines.extend(render_tree(node.left, depth + 1))
    lines.append(f"{indent}- Sinon")
    lines.extend(render_tree(node.right, depth + 1))
    return lines


def write_report(
    df: pl.DataFrame,
    reduced_df: pl.DataFrame,
    class_counts: dict[int, int],
    fraud_by_type: list[dict[str, Any]],
    tree_validation_confusion: dict[str, int],
    tree_validation_scores: dict[str, float],
    tree_test_confusion: dict[str, int],
    tree_test_scores: dict[str, float],
    tree_threshold: float,
    tree: TreeNode,
    logistic_validation_confusion: dict[str, int],
    logistic_validation_scores: dict[str, float],
    logistic_test_confusion: dict[str, int],
    logistic_test_scores: dict[str, float],
    logistic_threshold: float,
    logistic_model: LogisticModel,
    bayes_validation_scores: dict[str, float],
    bayes_test_confusion: dict[str, int],
    bayes_test_scores: dict[str, float],
    bayes_threshold: float,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    train_frauds: int,
    validation_frauds: int,
    test_frauds: int,
    split_train_max_step: int,
    split_validation_max_step: int,
    scaling_stats: dict[str, ScalingStat],
) -> None:
    fraud_rate = class_counts[1] / (class_counts[0] + class_counts[1])
    report = f"""# Rapport de detection de fraude

## Resume

- Lignes analysees: {df.height:,}
- Lignes retenues apres reduction: {reduced_df.height:,}
- Variables apres nettoyage / enrichissement: {df.width}
- Fraudes: {class_counts[1]:,}
- Taux de fraude: {fraud_rate:.4%}
- Seuil arbre de decision: {tree_threshold:.2f}
- Seuil regression logistique: {logistic_threshold:.2f}
- Seuil Naive Bayes gaussien: {bayes_threshold:.2f}

## Nettoyage

- Aucune valeur manquante detectee dans le dataset source.
- Colonnes textuelles identifiantes retirees du modele: `nameOrig`, `nameDest`.
- Variables derivees ajoutees: indicateurs de type, soldes nuls, compte origine vide apres transaction, incoherences de solde, ratios et logarithmes.
- Variables interdites au modele pour eviter la fuite d'information: `newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud` et toutes les variables derivees apres transaction.
- Variables numeriques du modele centrees-reduites apres clipping `q01-q99` calcule sur le train uniquement.
- Jeu nettoye exporte dans: `{CLEANED_PATH}`
- Jeu reduit pour l'entrainement exporte dans: `{REDUCED_PATH}`
- Jeu reduit centre-reduit exporte dans: `{SCALED_REDUCED_PATH}`

## Protocole anti fuite

- Entrainement: `step <= {split_train_max_step}` avec {train_rows:,} lignes et {train_frauds:,} fraudes.
- Validation: `{split_train_max_step} < step <= {split_validation_max_step}` avec {validation_rows:,} lignes et {validation_frauds:,} fraudes.
- Test: `step > {split_validation_max_step}` avec {test_rows:,} lignes et {test_frauds:,} fraudes.
- Le modele n'utilise que des variables disponibles avant ou au debut de la transaction.

## Schema

```mermaid
flowchart LR
    A[data.csv] --> B[Nettoyage]
    B --> C[data_fraud_cleaned.parquet]
    C --> D[Feature engineering]
    D --> E[Train stratife]
    D --> F[Validation]
    D --> G[Test]
    E --> H[Arbre de decision]
    F --> I[Choix du seuil]
    H --> J[Predictions]
    I --> J
    J --> K[Matrice de confusion]
    J --> L[Metriques]
```

## Comparaison des modeles

| Modele | Accuracy test | Precision test | Recall test | F1 test | Balanced accuracy test |
|---|---:|---:|---:|---:|---:|
| Arbre de decision | {tree_test_scores["accuracy"]:.4%} | {tree_test_scores["precision"]:.4%} | {tree_test_scores["recall"]:.4%} | {tree_test_scores["f1"]:.4%} | {tree_test_scores["balanced_accuracy"]:.4%} |
| Regression logistique | {logistic_test_scores["accuracy"]:.4%} | {logistic_test_scores["precision"]:.4%} | {logistic_test_scores["recall"]:.4%} | {logistic_test_scores["f1"]:.4%} | {logistic_test_scores["balanced_accuracy"]:.4%} |
| Naive Bayes gaussien | {bayes_test_scores["accuracy"]:.4%} | {bayes_test_scores["precision"]:.4%} | {bayes_test_scores["recall"]:.4%} | {bayes_test_scores["f1"]:.4%} | {bayes_test_scores["balanced_accuracy"]:.4%} |

## Metriques validation arbre

| Metrique | Valeur |
|---|---:|
| Accuracy | {tree_validation_scores["accuracy"]:.4%} |
| Precision | {tree_validation_scores["precision"]:.4%} |
| Recall | {tree_validation_scores["recall"]:.4%} |
| Specificity | {tree_validation_scores["specificity"]:.4%} |
| F1 | {tree_validation_scores["f1"]:.4%} |
| Balanced accuracy | {tree_validation_scores["balanced_accuracy"]:.4%} |

## Matrice de confusion test arbre

| Metrique | Valeur |
|---|---:|
| TN | {tree_test_confusion["tn"]:,} |
| FP | {tree_test_confusion["fp"]:,} |
| FN | {tree_test_confusion["fn"]:,} |
| TP | {tree_test_confusion["tp"]:,} |

## Metriques test arbre

| Metrique | Valeur |
|---|---:|
| Accuracy | {tree_test_scores["accuracy"]:.4%} |
| Precision | {tree_test_scores["precision"]:.4%} |
| Recall | {tree_test_scores["recall"]:.4%} |
| Specificity | {tree_test_scores["specificity"]:.4%} |
| F1 | {tree_test_scores["f1"]:.4%} |
| Balanced accuracy | {tree_test_scores["balanced_accuracy"]:.4%} |

## Metriques validation regression logistique

| Metrique | Valeur |
|---|---:|
| Accuracy | {logistic_validation_scores["accuracy"]:.4%} |
| Precision | {logistic_validation_scores["precision"]:.4%} |
| Recall | {logistic_validation_scores["recall"]:.4%} |
| Specificity | {logistic_validation_scores["specificity"]:.4%} |
| F1 | {logistic_validation_scores["f1"]:.4%} |
| Balanced accuracy | {logistic_validation_scores["balanced_accuracy"]:.4%} |

## Matrice de confusion test regression logistique

| Metrique | Valeur |
|---|---:|
| TN | {logistic_test_confusion["tn"]:,} |
| FP | {logistic_test_confusion["fp"]:,} |
| FN | {logistic_test_confusion["fn"]:,} |
| TP | {logistic_test_confusion["tp"]:,} |

## Metriques test regression logistique

| Metrique | Valeur |
|---|---:|
| Accuracy | {logistic_test_scores["accuracy"]:.4%} |
| Precision | {logistic_test_scores["precision"]:.4%} |
| Recall | {logistic_test_scores["recall"]:.4%} |
| Specificity | {logistic_test_scores["specificity"]:.4%} |
| F1 | {logistic_test_scores["f1"]:.4%} |
| Balanced accuracy | {logistic_test_scores["balanced_accuracy"]:.4%} |

## Matrice de confusion test Naive Bayes gaussien

| Metrique | Valeur |
|---|---:|
| TN | {bayes_test_confusion["tn"]:,} |
| FP | {bayes_test_confusion["fp"]:,} |
| FN | {bayes_test_confusion["fn"]:,} |
| TP | {bayes_test_confusion["tp"]:,} |

## Metriques test Naive Bayes gaussien

| Metrique | Valeur |
|---|---:|
| Accuracy | {bayes_test_scores["accuracy"]:.4%} |
| Precision | {bayes_test_scores["precision"]:.4%} |
| Recall | {bayes_test_scores["recall"]:.4%} |
| Specificity | {bayes_test_scores["specificity"]:.4%} |
| F1 | {bayes_test_scores["f1"]:.4%} |
| Balanced accuracy | {bayes_test_scores["balanced_accuracy"]:.4%} |

## Fraude par type

| Type | Normal | Fraude |
|---|---:|---:|
{chr(10).join(f'| {row["type"]} | {row["normal"]:,} | {row["fraud"]:,} |' for row in fraud_by_type)}

## Structure du modele

{chr(10).join(render_tree(tree))}

## Coefficients regression logistique

| Variable | Poids |
|---|---:|
{chr(10).join(f'| {feature} | {weight:.6f} |' for feature, weight in zip(logistic_model.features, logistic_model.weights))}
| bias | {logistic_model.bias:.6f} |

## Centrage reduction

| Variable | Clip bas | Clip haut | Moyenne train | Ecart-type train |
|---|---:|---:|---:|---:|
{chr(10).join(f'| {stat.feature} | {stat.clip_low:.4f} | {stat.clip_high:.4f} | {stat.mean:.4f} | {stat.std:.4f} |' for stat in scaling_stats.values())}

## Fichiers generes

- `{REPORT_PATH}`
- `{TREE_PATH}`
- `{LOGISTIC_PATH}`
- `{BAYES_PATH}`
- `{METRICS_PATH}`
- `{CONFUSION_CSV_PATH}`
- `{CLASS_SVG_PATH}`
- `{TYPE_SVG_PATH}`
- `{CONFUSION_SVG_PATH}`
- `{TYPE_RATE_SVG_PATH}`
- `{STEP_RATE_SVG_PATH}`
- `{AMOUNT_RATE_SVG_PATH}`
- `{SIGNAL_RATE_SVG_PATH}`
"""
    REPORT_PATH.write_text(report.replace(",", " "), encoding="utf-8")


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

    train_sample = train_df.filter(
        (pl.col("isFraud") == 1) | (pl.col("row_id") % 37 == 0)
    )

    thresholds = build_thresholds(train_sample)
    sample_rows = train_sample.select(MODEL_FEATURES + ["isFraud"]).iter_rows(named=True)
    tree = build_tree(list(sample_rows), thresholds, max_depth=4, min_samples=300, min_positives=8)
    logistic_model = fit_logistic_regression(train_df, MODEL_FEATURES, epochs=90, learning_rate=0.18)
    bayes_model = fit_gaussian_nb(train_df, MODEL_FEATURES)

    tree_threshold, tree_validation_scores, tree_validation_confusion = tune_tree_threshold(tree, validation_df)
    tree_test_confusion, tree_test_scores = evaluate(tree, test_df, tree_threshold)
    logistic_threshold, logistic_validation_scores, logistic_validation_confusion = tune_logistic_threshold(
        logistic_model, validation_df
    )
    logistic_test_confusion, logistic_test_scores = evaluate_logistic(
        logistic_model, test_df, logistic_threshold
    )
    bayes_threshold, bayes_validation_scores, bayes_validation_confusion = tune_bayes_threshold(
        bayes_model, validation_df
    )
    bayes_test_confusion, bayes_test_scores = evaluate_bayes(
        bayes_model, test_df, bayes_threshold
    )

    class_rows = df.group_by("isFraud").len().sort("isFraud").to_dicts()
    reduced_class_rows = reduced_df.group_by("isFraud").len().sort("isFraud").to_dicts()
    class_counts = {int(row["isFraud"]): int(row["len"]) for row in reduced_class_rows}

    fraud_by_type_rows = (
        reduced_df.group_by(["type", "isFraud"])
        .len()
        .sort(["type", "isFraud"])
        .to_dicts()
    )
    fraud_by_type: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, int]] = {}
    for row in fraud_by_type_rows:
        current = by_type.setdefault(row["type"], {"type": row["type"], "normal": 0, "fraud": 0})
        if int(row["isFraud"]) == 1:
            current["fraud"] = int(row["len"])
        else:
            current["normal"] = int(row["len"])
    fraud_by_type = list(by_type.values())

    fraud_rate_by_type_rows = (
        df.group_by("type")
        .agg(
            [
                pl.len().alias("transactions"),
                pl.col("isFraud").sum().alias("frauds"),
                pl.col("isFraud").mean().alias("fraud_rate"),
            ]
        )
        .sort("fraud_rate", descending=True)
        .to_dicts()
    )

    fraud_rate_by_period_rows = (
        df.with_columns((((pl.col("step") - 1) // 24) + 1).alias("period"))
        .group_by("period")
        .agg(
            [
                pl.len().alias("transactions"),
                pl.col("isFraud").sum().alias("frauds"),
                pl.col("isFraud").mean().alias("fraud_rate"),
            ]
        )
        .sort("period")
        .to_dicts()
    )

    amount_bucket_expr = (
        pl.when(pl.col("amount") < 1_000)
        .then(pl.lit("<1k"))
        .when(pl.col("amount") < 10_000)
        .then(pl.lit("1k-10k"))
        .when(pl.col("amount") < 100_000)
        .then(pl.lit("10k-100k"))
        .when(pl.col("amount") < 1_000_000)
        .then(pl.lit("100k-1M"))
        .when(pl.col("amount") < 10_000_000)
        .then(pl.lit("1M-10M"))
        .otherwise(pl.lit("10M+"))
        .alias("amount_bucket")
    )
    amount_bucket_order = {
        "<1k": 0,
        "1k-10k": 1,
        "10k-100k": 2,
        "100k-1M": 3,
        "1M-10M": 4,
        "10M+": 5,
    }
    fraud_rate_by_amount_rows = (
        df.with_columns(amount_bucket_expr)
        .group_by("amount_bucket")
        .agg(
            [
                pl.len().alias("transactions"),
                pl.col("isFraud").sum().alias("frauds"),
                pl.col("isFraud").mean().alias("fraud_rate"),
            ]
        )
        .with_columns(pl.col("amount_bucket").replace(amount_bucket_order).alias("order"))
        .sort("order")
        .drop("order")
        .to_dicts()
    )

    signal_features = [
        ("emptied_origin", "Orig videe"),
        ("dest_zero", "Dest solde 0"),
        ("dest_untouched", "Dest intacte"),
        ("orig_zero", "Orig solde 0"),
        ("balance_error_dest", "Erreur dest"),
        ("balance_error_orig", "Erreur orig"),
        ("flagged_fraud", "Flag systeme"),
    ]
    signal_comparison_rows: list[dict[str, Any]] = []
    for feature, label in signal_features:
        rates = (
            df.group_by("isFraud")
            .agg(pl.col(feature).mean().alias("rate"))
            .sort("isFraud")
            .to_dicts()
        )
        normal_rate = next(float(row["rate"]) for row in rates if int(row["isFraud"]) == 0)
        fraud_rate = next(float(row["rate"]) for row in rates if int(row["isFraud"]) == 1)
        signal_comparison_rows.append(
            {"label": label, "normal_rate": normal_rate, "fraud_rate": fraud_rate}
        )

    pl.DataFrame(
        {
            "prediction_0": [tree_test_confusion["tn"], tree_test_confusion["fn"]],
            "prediction_1": [tree_test_confusion["fp"], tree_test_confusion["tp"]],
        },
        schema={"prediction_0": pl.Int64, "prediction_1": pl.Int64},
    ).write_csv(CONFUSION_CSV_PATH)

    build_svg_bar_chart(
        [("Normal", class_counts[0]), ("Fraude", class_counts[1])],
        CLASS_SVG_PATH,
        "Distribution des classes",
        "#5b8c5a",
    )
    build_svg_grouped_chart(
        [row["type"] for row in fraud_by_type],
        [int(row["normal"]) for row in fraud_by_type],
        [int(row["fraud"]) for row in fraud_by_type],
        TYPE_SVG_PATH,
        "Fraude par type de transaction",
    )
    build_svg_confusion(tree_test_confusion, CONFUSION_SVG_PATH)
    build_svg_rate_chart(
        [(str(row["type"]), float(row["fraud_rate"])) for row in fraud_rate_by_type_rows],
        TYPE_RATE_SVG_PATH,
        "Taux de fraude par type de transaction",
        "#c8553d",
    )
    build_svg_rate_chart(
        [
            (f"P{int(row['period'])}", float(row["fraud_rate"]))
            for row in fraud_rate_by_period_rows
        ],
        STEP_RATE_SVG_PATH,
        "Taux de fraude par periode de 24 steps",
        "#d08c60",
    )
    build_svg_rate_chart(
        [
            (str(row["amount_bucket"]), float(row["fraud_rate"]))
            for row in fraud_rate_by_amount_rows
        ],
        AMOUNT_RATE_SVG_PATH,
        "Taux de fraude par tranche de montant",
        "#8b6f47",
    )
    build_svg_grouped_rate_chart(
        [str(row["label"]) for row in signal_comparison_rows],
        [float(row["normal_rate"]) for row in signal_comparison_rows],
        [float(row["fraud_rate"]) for row in signal_comparison_rows],
        SIGNAL_RATE_SVG_PATH,
        "Comparaison des signaux entre normal et fraude",
    )

    TREE_PATH.write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")
    LOGISTIC_PATH.write_text(json.dumps(asdict(logistic_model), indent=2), encoding="utf-8")
    BAYES_PATH.write_text(json.dumps(asdict(bayes_model), indent=2), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(
            {
                "tree": {
                    "threshold": tree_threshold,
                    "validation_confusion": tree_validation_confusion,
                    "validation_scores": tree_validation_scores,
                    "test_confusion": tree_test_confusion,
                    "test_scores": tree_test_scores,
                },
                "logistic_regression": {
                    "threshold": logistic_threshold,
                    "validation_confusion": logistic_validation_confusion,
                    "validation_scores": logistic_validation_scores,
                    "test_confusion": logistic_test_confusion,
                    "test_scores": logistic_test_scores,
                },
                "gaussian_nb": {
                    "threshold": bayes_threshold,
                    "validation_confusion": bayes_validation_confusion,
                    "validation_scores": bayes_validation_scores,
                    "test_confusion": bayes_test_confusion,
                    "test_scores": bayes_test_scores,
                },
                "scaling_stats": {
                    feature: asdict(stat) for feature, stat in scaling_stats.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        df=df,
        reduced_df=reduced_df,
        class_counts=class_counts,
        fraud_by_type=fraud_by_type,
        tree_validation_confusion=tree_validation_confusion,
        tree_validation_scores=tree_validation_scores,
        tree_test_confusion=tree_test_confusion,
        tree_test_scores=tree_test_scores,
        tree_threshold=tree_threshold,
        tree=tree,
        logistic_validation_confusion=logistic_validation_confusion,
        logistic_validation_scores=logistic_validation_scores,
        logistic_test_confusion=logistic_test_confusion,
        logistic_test_scores=logistic_test_scores,
        logistic_threshold=logistic_threshold,
        logistic_model=logistic_model,
        bayes_validation_scores=bayes_validation_scores,
        bayes_test_confusion=bayes_test_confusion,
        bayes_test_scores=bayes_test_scores,
        bayes_threshold=bayes_threshold,
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
